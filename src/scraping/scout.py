# ============================================================================
# SCOUT ROBUSTE POUR DÉTECTION DE BALISES
# ============================================================================

class RobustProductReviewScout:
    """
    Scout ultra-robuste pour détecter automatiquement les balises de produits et reviews
    Avec gestion d'erreurs complète et fallbacks multiples
    """
    
    def __init__(self):
        self.driver = None
        self.detected_selectors = {}
        self.base_selectors = {
            'amazon': {
                'products': {
                    'containers': [
                        '[data-component-type="s-search-result"]',
                        '.s-result-item',
                        '.s-widget-container .s-card-container',
                        '[data-asin]:not([data-asin=""])'
                    ],
                    'titles': [
                        'h2 span',
                        'h2 a span',
                        '.s-size-mini span',
                        '.a-size-base-plus',
                        '[data-cy="title-recipe-title"]'
                    ],
                    'urls': [
                        'h2 a',
                        '.a-link-normal',
                        'a[href*="/dp/"]',
                        'a[href*="/gp/product/"]'
                    ],
                    'prices': [
                        '.a-price .a-offscreen',
                        '.a-price-whole',
                        '.a-price-range .a-offscreen',
                        '.a-price-symbol + .a-price-whole'
                    ],
                    'ratings': [
                        '.a-icon-alt',
                        '.a-star-mini .a-icon-alt',
                        'span[aria-label*="stars"]'
                    ]
                },
                'reviews': {
                    'containers': [
                        '[data-hook="review"]',
                        '.review',
                        '.cr-original-review-content',
                        '.reviewText'
                    ],
                    'titles': [
                        '[data-hook="review-title"] span',
                        '.review-title',
                        '.cr-original-review-content .review-title'
                    ],
                    'texts': [
                        '[data-hook="review-body"] span',
                        '.review-text',
                        '.cr-original-review-content .review-text',
                        '.reviewText'
                    ],
                    'ratings': [
                        '[data-hook="review-star-rating"] .a-icon-alt',
                        '.review-rating .a-icon-alt',
                        '.cr-original-review-content .a-icon-alt'
                    ],
                    'authors': [
                        '.a-profile-name',
                        '.review-author',
                        '.cr-original-review-content .author'
                    ],
                    'dates': [
                        '[data-hook="review-date"]',
                        '.review-date',
                        '.cr-original-review-content .review-date'
                    ]
                }
            },
            'ebay': {
                'products': {
                    'containers': [
                        '.s-item',
                        '.srp-results .s-item',
                        '.b-listing__wrap'
                    ],
                    'titles': [
                        '.s-item__title',
                        '.it-ttl',
                        '.b-listing__title'
                    ],
                    'urls': [
                        '.s-item__link',
                        '.it-ttl a',
                        '.b-listing__title a'
                    ],
                    'prices': [
                        '.s-item__price',
                        '.notranslate',
                        '.b-listing__price'
                    ]
                },
                'reviews': {
                    'containers': [
                        '.review-item',
                        '.ebay-review',
                        '.reviews .review'
                    ],
                    'texts': [
                        '.review-item-content',
                        '.ebay-review-text',
                        '.review-content'
                    ]
                }
            }
        }
    
    def setup_robust_driver(self, headless=True, timeout=30):
        """Configuration driver ultra-robuste avec fallbacks"""
        
        max_attempts = 3
        
        for attempt in range(max_attempts):
            try:
                print(f"🔧 Tentative {attempt + 1}/{max_attempts} - Setup driver scout...")
                
                # Fermer le driver existant si nécessaire
                if self.driver:
                    try:
                        self.driver.quit()
                    except:
                        pass
                    self.driver = None
                
                # Options ultra-sûres
                options = self._create_safe_options(headless)
                
                # Création driver avec timeout
                import undetected_chromedriver as uc
                
                self.driver = uc.Chrome(
                    options=options,
                    version_main=None,
                    headless=headless,
                    use_subprocess=False,
                    log_level=3
                )
                
                # Configuration des timeouts
                self.driver.set_page_load_timeout(timeout)
                self.driver.implicitly_wait(10)
                
                # Test de fonctionnement
                self.driver.get("data:text/html,<html><body><h1>Test Scout</h1></body></html>")
                
                print("✅ Driver scout initialisé avec succès!")
                return True
                
            except Exception as e:
                print(f"❌ Tentative {attempt + 1} échouée: {str(e)[:100]}...")
                if attempt < max_attempts - 1:
                    time.sleep(3)
                else:
                    print("❌ Impossible de créer le driver scout")
                    return False
        
        return False
    
    def _create_safe_options(self, headless=True):
        """Crée des options Chrome ultra-sûres"""
        
        import undetected_chromedriver as uc
        
        options = uc.ChromeOptions()
        
        # Arguments de base seulement
        safe_args = [
            '--no-sandbox',
            '--disable-dev-shm-usage',
            '--disable-gpu',
            '--disable-web-security',
            '--disable-features=VizDisplayCompositor',
            '--window-size=1920,1080',
            '--start-maximized'
        ]
        
        for arg in safe_args:
            options.add_argument(arg)
        
        if headless:
            options.add_argument('--headless=new')
        
        # User agent aléatoire
        try:
            user_agent = random.choice(REALISTIC_USER_AGENTS)
            options.add_argument(f'--user-agent={user_agent}')
        except:
            options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
        
        # Préférences minimales
        prefs = {
            "profile.default_content_setting_values.notifications": 2,
            "profile.default_content_settings.popups": 0
        }
        options.add_experimental_option("prefs", prefs)
        
        return options
    
    def detect_site_selectors(self, site_url, search_term="laptop", max_retries=3):
        """
        Détecte automatiquement tous les sélecteurs pour un site
        """
        
        if not self.driver:
            print("❌ Driver non initialisé")
            return {}
        
        site_type = 'amazon' if 'amazon' in site_url else 'ebay' if 'ebay' in site_url else 'unknown'
        
        if site_type == 'unknown':
            print(f"❌ Site non supporté: {site_url}")
            return {}
        
        print(f"🔍 Détection des sélecteurs pour {site_type}...")
        
        detected = {
            'site': site_type,
            'products': {},
            'reviews': {}
        }
        
        try:
            # Phase 1: Détection sélecteurs produits
            product_selectors = self._detect_product_selectors(site_url, search_term, site_type)
            detected['products'] = product_selectors
            
            if product_selectors:
                print(f"✅ Sélecteurs produits détectés: {len(product_selectors)}")
                
                # Phase 2: Détection sélecteurs reviews
                review_selectors = self._detect_review_selectors(site_type, product_selectors)
                detected['reviews'] = review_selectors
                
                if review_selectors:
                    print(f"✅ Sélecteurs reviews détectés: {len(review_selectors)}")
                else:
                    print("⚠️ Sélecteurs reviews non détectés, utilisation des défauts")
                    detected['reviews'] = self.base_selectors[site_type]['reviews']
            else:
                print("❌ Impossible de détecter les sélecteurs produits")
                return {}
            
            # Sauvegarde des sélecteurs
            self._save_detected_selectors(detected, site_type)
            
            return detected
            
        except Exception as e:
            print(f"❌ Erreur détection: {e}")
            return {}
    
    def _detect_product_selectors(self, site_url, search_term, site_type):
        """Détecte les sélecteurs de produits avec tests multiples"""
        
        try:
            # Construire URL de recherche
            if site_type == 'amazon':
                search_url = f"https://www.amazon.com/s?k={search_term.replace(' ', '+')}"
            elif site_type == 'ebay':
                search_url = f"https://www.ebay.com/sch/i.html?_nkw={search_term.replace(' ', '+')}"
            else:
                return {}
            
            print(f"🌐 Navigation vers: {search_url}")
            self.driver.get(search_url)
            
            # Attendre le chargement
            time.sleep(5)
            
            # Tester les sélecteurs de conteneurs
            selectors = {}
            base_selectors = self.base_selectors[site_type]['products']
            
            # Test conteneurs de produits
            for selector in base_selectors['containers']:
                try:
                    elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    if len(elements) >= 3:  # Au moins 3 produits
                        selectors['container'] = selector
                        print(f"✅ Conteneur: {selector} ({len(elements)} éléments)")
                        break
                except:
                    continue
            
            if not selectors.get('container'):
                print("❌ Aucun conteneur de produit trouvé")
                return {}
            
            # Test autres sélecteurs dans le contexte du conteneur
            container_elements = self.driver.find_elements(By.CSS_SELECTOR, selectors['container'])
            
            if container_elements:
                first_container = container_elements[0]
                
                # Test titres
                for title_selector in base_selectors['titles']:
                    try:
                        title_elem = first_container.find_element(By.CSS_SELECTOR, title_selector)
                        if title_elem.text.strip():
                            selectors['title'] = title_selector
                            print(f"✅ Titre: {title_selector}")
                            break
                    except:
                        continue
                
                # Test URLs
                for url_selector in base_selectors['urls']:
                    try:
                        url_elem = first_container.find_element(By.CSS_SELECTOR, url_selector)
                        href = url_elem.get_attribute('href')
                        if href and ('amazon.com' in href or 'ebay.com' in href):
                            selectors['url'] = url_selector
                            print(f"✅ URL: {url_selector}")
                            break
                    except:
                        continue
                
                # Test prix
                for price_selector in base_selectors['prices']:
                    try:
                        price_elem = first_container.find_element(By.CSS_SELECTOR, price_selector)
                        if price_elem.text.strip() and ('$' in price_elem.text or '€' in price_elem.text):
                            selectors['price'] = price_selector
                            print(f"✅ Prix: {price_selector}")
                            break
                    except:
                        continue
                
                # Test ratings
                for rating_selector in base_selectors['ratings']:
                    try:
                        rating_elem = first_container.find_element(By.CSS_SELECTOR, rating_selector)
                        rating_text = rating_elem.get_attribute('textContent') or rating_elem.text
                        if rating_text and ('star' in rating_text.lower() or 'étoile' in rating_text.lower()):
                            selectors['rating'] = rating_selector
                            print(f"✅ Rating: {rating_selector}")
                            break
                    except:
                        continue
            
            return selectors
            
        except Exception as e:
            print(f"❌ Erreur détection produits: {e}")
            return {}
    
    def _detect_review_selectors(self, site_type, product_selectors):
        """Détecte les sélecteurs de reviews en naviguant vers un produit"""
        
        try:
            # Trouver un lien produit
            if not product_selectors.get('url'):
                print("❌ Pas de sélecteur URL disponible")
                return {}
            
            # Récupérer le premier lien produit
            product_links = self.driver.find_elements(By.CSS_SELECTOR, product_selectors['url'])
            
            if not product_links:
                print("❌ Aucun lien produit trouvé")
                return {}
            
            product_url = product_links[0].get_attribute('href')
            print(f"🔗 Test reviews sur: {product_url[:80]}...")
            
            # Naviguer vers le produit
            self.driver.get(product_url)
            time.sleep(3)
            
            # Chercher les reviews sur la page produit ou naviguer vers la page reviews
            review_selectors = {}
            base_selectors = self.base_selectors[site_type]['reviews']
            
            # D'abord, chercher un lien vers les reviews
            review_link_selectors = [
                'a[href*="customer-reviews"]',
                'a[href*="reviews"]',
                'a[href*="review"]',
                '.cr-widget-ACR a'
            ]
            
            review_page_found = False
            for link_selector in review_link_selectors:
                try:
                    review_links = self.driver.find_elements(By.CSS_SELECTOR, link_selector)
                    for link in review_links:
                        href = link.get_attribute('href')
                        if href and 'review' in href:
                            print(f"🔗 Navigation vers page reviews: {href[:80]}...")
                            self.driver.get(href)
                            time.sleep(3)
                            review_page_found = True
                            break
                    if review_page_found:
                        break
                except:
                    continue
            
            # Tester les sélecteurs de reviews
            # Test conteneurs
            for container_selector in base_selectors['containers']:
                try:
                    review_elements = self.driver.find_elements(By.CSS_SELECTOR, container_selector)
                    if len(review_elements) >= 2:  # Au moins 2 reviews
                        review_selectors['container'] = container_selector
                        print(f"✅ Conteneur reviews: {container_selector} ({len(review_elements)} reviews)")
                        break
                except:
                    continue
            
            if review_selectors.get('container'):
                # Tester les autres sélecteurs dans le contexte
                review_containers = self.driver.find_elements(By.CSS_SELECTOR, review_selectors['container'])
                
                if review_containers:
                    first_review = review_containers[0]
                    
                    # Test texte de review
                    for text_selector in base_selectors['texts']:
                        try:
                            text_elem = first_review.find_element(By.CSS_SELECTOR, text_selector)
                            if text_elem.text.strip() and len(text_elem.text.strip()) > 20:
                                review_selectors['text'] = text_selector
                                print(f"✅ Texte review: {text_selector}")
                                break
                        except:
                            continue
                    
                    # Test titre de review
                    for title_selector in base_selectors['titles']:
                        try:
                            title_elem = first_review.find_element(By.CSS_SELECTOR, title_selector)
                            if title_elem.text.strip():
                                review_selectors['title'] = title_selector
                                print(f"✅ Titre review: {title_selector}")
                                break
                        except:
                            continue
                    
                    # Test rating
                    for rating_selector in base_selectors['ratings']:
                        try:
                            rating_elem = first_review.find_element(By.CSS_SELECTOR, rating_selector)
                            rating_text = rating_elem.get_attribute('textContent') or rating_elem.text
                            if rating_text and ('star' in rating_text.lower() or 'étoile' in rating_text.lower()):
                                review_selectors['rating'] = rating_selector
                                print(f"✅ Rating review: {rating_selector}")
                                break
                        except:
                            continue
                    
                    # Test auteur
                    for author_selector in base_selectors['authors']:
                        try:
                            author_elem = first_review.find_element(By.CSS_SELECTOR, author_selector)
                            if author_elem.text.strip():
                                review_selectors['author'] = author_selector
                                print(f"✅ Auteur review: {author_selector}")
                                break
                        except:
                            continue
                    
                    # Test date
                    for date_selector in base_selectors['dates']:
                        try:
                            date_elem = first_review.find_element(By.CSS_SELECTOR, date_selector)
                            if date_elem.text.strip():
                                review_selectors['date'] = date_selector
                                print(f"✅ Date review: {date_selector}")
                                break
                        except:
                            continue
            
            return review_selectors
            
        except Exception as e:
            print(f"❌ Erreur détection reviews: {e}")
            return {}
    
    def _save_detected_selectors(self, selectors, site_type):
        """Sauvegarde les sélecteurs détectés"""
        
        try:
            import os
            import json
            
            config_dir = "../config"
            os.makedirs(config_dir, exist_ok=True)
            
            filename = f"{config_dir}/detected_selectors_{site_type}.json"
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(selectors, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Sélecteurs sauvegardés: {filename}")
            return True
            
        except Exception as e:
            print(f"❌ Erreur sauvegarde: {e}")
            return False
    
    def close(self):
        """Ferme proprement le driver"""
        if self.driver:
            try:
                self.driver.quit()
                print("✅ Driver scout fermé")
            except:
                pass
            self.driver = None

print("✅ Scout robuste créé!")