# ============================================================================
# SCRAPER ROBUSTE POUR REVIEWS DE PRODUITS
# ============================================================================

class RobustProductReviewScraper:
    """
    Scraper ultra-robuste pour récupérer les reviews de produits
    Utilise les sélecteurs détectés par le scout
    """
    
    def __init__(self, selectors_file=None):
        self.driver = None
        self.selectors = {}
        self.scraped_data = []
        self.session_stats = {
            'products_processed': 0,
            'reviews_collected': 0,
            'errors': 0,
            'start_time': None
        }
        
        if selectors_file:
            self.load_selectors(selectors_file)
    
    def load_selectors(self, filename):
        """Charge les sélecteurs depuis un fichier JSON"""
        try:
            import json
            with open(filename, 'r', encoding='utf-8') as f:
                self.selectors = json.load(f)
            print(f"✅ Sélecteurs chargés: {filename}")
            return True
        except Exception as e:
            print(f"❌ Erreur chargement sélecteurs: {e}")
            return False
    
    def setup_robust_driver(self, headless=False, timeout=60):
        """Configuration driver ultra-robuste pour scraping"""
        
        max_attempts = 3
        
        for attempt in range(max_attempts):
            try:
                print(f"🚀 Tentative {attempt + 1}/{max_attempts} - Setup driver scraper...")
                
                # Fermer le driver existant
                if self.driver:
                    try:
                        self.driver.quit()
                    except:
                        pass
                    self.driver = None
                
                # Options anti-détection
                options = self._create_stealth_options(headless)
                
                # Création du driver
                import undetected_chromedriver as uc
                
                self.driver = uc.Chrome(
                    options=options,
                    version_main=None,
                    headless=headless,
                    use_subprocess=False,
                    log_level=3
                )
                
                # Configuration timeouts
                self.driver.set_page_load_timeout(timeout)
                self.driver.implicitly_wait(15)
                
                # Scripts anti-détection
                self._inject_stealth_scripts()
                
                # Test fonctionnement
                self.driver.get("data:text/html,<html><body><h1>Scraper Ready</h1></body></html>")
                
                print("✅ Driver scraper prêt!")
                print(f"🎭 User-Agent: {self.driver.execute_script('return navigator.userAgent;')[:80]}...")
                
                return True
                
            except Exception as e:
                print(f"❌ Tentative {attempt + 1} échouée: {str(e)[:100]}...")
                if attempt < max_attempts - 1:
                    time.sleep(5)
                else:
                    print("❌ Impossible de créer le driver scraper")
                    return False
        
        return False
    
    def _create_stealth_options(self, headless=True):
        """Crée des options Chrome furtives"""
        
        import undetected_chromedriver as uc
        
        options = uc.ChromeOptions()
        
        # Arguments furtifs
        stealth_args = [
            '--no-sandbox',
            '--disable-dev-shm-usage',
            '--disable-gpu',
            '--disable-web-security',
            '--disable-features=VizDisplayCompositor',
            '--disable-blink-features=AutomationControlled',
            '--disable-extensions',
            '--no-first-run',
            '--no-default-browser-check',
            '--disable-default-apps',
            '--disable-background-timer-throttling',
            '--disable-backgrounding-occluded-windows',
            '--disable-renderer-backgrounding',
            '--window-size=1920,1080',
            '--start-maximized'
        ]
        
        for arg in stealth_args:
            options.add_argument(arg)
        
        if headless:
            options.add_argument('--headless=new')
        
        # User agent aléatoire réaliste
        try:
            user_agent = random.choice(REALISTIC_USER_AGENTS)
            options.add_argument(f'--user-agent={user_agent}')
        except:
            options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
        
        # Préférences furtives
        prefs = {
            "profile.default_content_setting_values.notifications": 2,
            "profile.default_content_settings.popups": 0,
            "profile.managed_default_content_settings.images": 1,  # Charger les images
            "profile.default_content_setting_values.plugins": 1,
            "profile.content_settings.plugin_whitelist.adobe-flash-player": 1,
            "profile.content_settings.exceptions.plugins.*,*.per_resource.adobe-flash-player": 1
        }
        options.add_experimental_option("prefs", prefs)
        
        return options
    
    def _inject_stealth_scripts(self):
        """Injecte des scripts anti-détection"""
        try:
            stealth_script = '''
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined,
                });
                
                Object.defineProperty(navigator, 'plugins', {
                    get: () => [1, 2, 3, 4, 5],
                });
                
                Object.defineProperty(navigator, 'languages', {
                    get: () => ['en-US', 'en'],
                });
                
                window.chrome = {
                    runtime: {},
                };
                
                Object.defineProperty(navigator, 'permissions', {
                    get: () => ({
                        query: () => Promise.resolve({ state: 'granted' }),
                    }),
                });
            '''
            
            self.driver.execute_cdp_cmd('Page.addScriptToEvaluateOnNewDocument', {
                'source': stealth_script
            })
            
        except Exception as e:
            print(f"⚠️ Scripts anti-détection non injectés: {e}")
    
    def scrape_category_reviews(self, category, site='amazon', max_products=10, reviews_per_rating=50):
        """
        Scrape principal pour récupérer les reviews d'une catégorie
        """
        
        if not self.driver:
            print("❌ Driver non initialisé")
            return pd.DataFrame()
        
        if not self.selectors:
            print("❌ Sélecteurs non chargés")
            return pd.DataFrame()
        
        print("="*80)
        print("🎯 DÉBUT DU SCRAPING ROBUSTE")
        print("="*80)
        print(f"📦 Catégorie: {category}")
        print(f"🌐 Site: {site}")
        print(f"📊 Produits max: {max_products}")
        print(f"⭐ Reviews par note: {reviews_per_rating}")
        print()
        
        # Initialiser les stats
        self.session_stats['start_time'] = time.time()
        
        try:
            # Phase 1: Récupérer la liste des produits
            products = self._get_products_list(category, site, max_products)
            
            if not products:
                print("❌ Aucun produit trouvé")
                return pd.DataFrame()
            
            print(f"✅ {len(products)} produits trouvés")
            
            # Phase 2: Scraper les reviews de chaque produit
            all_reviews = []
            
            for i, product in enumerate(products, 1):
                print(f"\n📦 PRODUIT {i}/{len(products)}")
                print(f"🏷️ {product['title'][:60]}...")
                
                try:
                    product_reviews = self._scrape_product_reviews(
                        product, 
                        reviews_per_rating,
                        max_pages=5
                    )
                    
                    if product_reviews:
                        all_reviews.extend(product_reviews)
                        self.session_stats['reviews_collected'] += len(product_reviews)
                        print(f"✅ {len(product_reviews)} reviews récupérées")
                    else:
                        print("⚠️ Aucune review trouvée")
                    
                    self.session_stats['products_processed'] += 1
                    
                    # Délai humain entre produits
                    delay = random.uniform(3, 8)
                    print(f"⏳ Délai: {delay:.1f}s...")
                    time.sleep(delay)
                    
                except Exception as e:
                    print(f"❌ Erreur produit {i}: {e}")
                    self.session_stats['errors'] += 1
                    continue
            
            # Phase 3: Créer et nettoyer le DataFrame
            if all_reviews:
                df = pd.DataFrame(all_reviews)
                df = self._clean_review_data(df)
                
                # Stats finales
                duration = time.time() - self.session_stats['start_time']
                print("\n" + "="*80)
                print("📊 SCRAPING TERMINÉ - STATISTIQUES")
                print("="*80)
                print(f"⏱️ Durée: {duration/60:.1f} minutes")
                print(f"📦 Produits traités: {self.session_stats['products_processed']}")
                print(f"📝 Reviews récupérées: {self.session_stats['reviews_collected']}")
                print(f"❌ Erreurs: {self.session_stats['errors']}")
                print(f"📈 Taux de succès: {(1-self.session_stats['errors']/max(1,len(products)))*100:.1f}%")
                
                return df
            else:
                print("❌ Aucune review récupérée")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"❌ Erreur scraping global: {e}")
            return pd.DataFrame()
    
    def _get_products_list(self, category, site, max_products):
        """Récupère la liste des produits à scraper"""
        
        try:
            # URL de recherche
            if site == 'amazon':
                search_url = f"https://www.amazon.com/s?k={category.replace(' ', '+')}"
            elif site == 'ebay':
                search_url = f"https://www.ebay.com/sch/i.html?_nkw={category.replace(' ', '+')}"
            else:
                print(f"❌ Site non supporté: {site}")
                return []
            
            print(f"🔍 Recherche: {search_url}")
            self.driver.get(search_url)
            
            # Attendre le chargement
            time.sleep(5)
            
            # Sélecteurs pour ce site
            site_selectors = self.selectors.get('products', {})
            
            if not site_selectors:
                print("❌ Sélecteurs produits non disponibles")
                return []
            
            # Récupérer les conteneurs de produits
            container_selector = site_selectors.get('container')
            if not container_selector:
                print("❌ Sélecteur conteneur manquant")
                return []
            
            containers = self.driver.find_elements(By.CSS_SELECTOR, container_selector)
            print(f"📦 {len(containers)} conteneurs trouvés")
            
            products = []
            
            for i, container in enumerate(containers[:max_products]):
                try:
                    product_data = self._extract_product_info(container, site_selectors, category)
                    
                    if product_data and product_data.get('url'):
                        products.append(product_data)
                        print(f"✅ Produit {len(products)}: {product_data['title'][:40]}...")
                    
                except Exception as e:
                    print(f"⚠️ Produit {i+1} ignoré: {e}")
                    continue
            
            return products
            
        except Exception as e:
            print(f"❌ Erreur récupération produits: {e}")
            return []
    
    def _extract_product_info(self, container, selectors, category):
        """Extrait les infos d'un produit depuis son conteneur"""
        
        product_data = {
            'category': category,
            'scraped_at': datetime.now().isoformat()
        }
        
        # Titre
        try:
            if selectors.get('title'):
                title_elem = container.find_element(By.CSS_SELECTOR, selectors['title'])
                product_data['title'] = title_elem.text.strip()
        except:
            product_data['title'] = 'Titre non trouvé'
        
        # URL
        try:
            if selectors.get('url'):
                url_elem = container.find_element(By.CSS_SELECTOR, selectors['url'])
                product_data['url'] = url_elem.get_attribute('href')
        except:
            product_data['url'] = None
        
        # Prix
        try:
            if selectors.get('price'):
                price_elem = container.find_element(By.CSS_SELECTOR, selectors['price'])
                product_data['price'] = price_elem.text.strip()
        except:
            product_data['price'] = 'N/A'
        
        # Rating
        try:
            if selectors.get('rating'):
                rating_elem = container.find_element(By.CSS_SELECTOR, selectors['rating'])
                rating_text = rating_elem.get_attribute('textContent') or rating_elem.text
                product_data['rating'] = rating_text.strip()
        except:
            product_data['rating'] = 'N/A'
        
        return product_data
    
    def _scrape_product_reviews(self, product, reviews_per_rating, max_pages=5):
        """Scrape les reviews d'un produit spécifique"""
        
        try:
            # Naviguer vers le produit
            self.driver.get(product['url'])
            time.sleep(3)
            
            # Trouver la page des reviews
            reviews_url = self._find_reviews_page(product['url'])
            
            if reviews_url:
                self.driver.get(reviews_url)
                time.sleep(3)
            else:
                print("⚠️ Page reviews non trouvée, tentative sur page produit")
            
            # Récupérer les reviews
            all_reviews = []
            
            # Sélecteurs reviews
            review_selectors = self.selectors.get('reviews', {})
            
            if not review_selectors:
                print("❌ Sélecteurs reviews non disponibles")
                return []
            
            # Scraper les reviews page par page
            for page in range(max_pages):
                try:
                    page_reviews = self._extract_reviews_from_page(product, review_selectors)
                    
                    if page_reviews:
                        all_reviews.extend(page_reviews)
                        print(f"📝 Page {page+1}: {len(page_reviews)} reviews")
                        
                        # Essayer de passer à la page suivante
                        if not self._go_to_next_page():
                            print("📄 Plus de pages disponibles")
                            break
                        
                        time.sleep(random.uniform(2, 4))
                    else:
                        print(f"📄 Page {page+1}: aucune review")
                        break
                        
                except Exception as e:
                    print(f"❌ Erreur page {page+1}: {e}")
                    break
            
            # Limiter le nombre de reviews si nécessaire
            if len(all_reviews) > reviews_per_rating * 5:  # 5 notes possibles
                all_reviews = all_reviews[:reviews_per_rating * 5]
            
            return all_reviews
            
        except Exception as e:
            print(f"❌ Erreur scraping reviews produit: {e}")
            return []
    
    def _find_reviews_page(self, product_url):
        """Trouve l'URL de la page des reviews"""
        
        try:
            # Sélecteurs de liens vers reviews
            review_link_selectors = [
                'a[href*="customer-reviews"]',
                'a[href*="reviews"]',
                '[data-hook="see-all-reviews-link-foot"]',
                '.cr-widget-ACR a'
            ]
            
            for selector in review_link_selectors:
                try:
                    links = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    for link in links:
                        href = link.get_attribute('href')
                        if href and 'review' in href:
                            return href
                except:
                    continue
            
            # Si aucun lien trouvé, construire l'URL pour Amazon
            if 'amazon.com' in product_url:
                import re
                asin_match = re.search(r'/dp/([A-Z0-9]{10})', product_url)
                if asin_match:
                    asin = asin_match.group(1)
                    return f"https://www.amazon.com/product-reviews/{asin}"
            
            return None
            
        except Exception as e:
            print(f"⚠️ Erreur recherche page reviews: {e}")
            return None
    
    def _extract_reviews_from_page(self, product, review_selectors):
        """Extrait les reviews de la page actuelle"""
        
        reviews = []
        
        try:
            # Récupérer les conteneurs de reviews
            container_selector = review_selectors.get('container')
            if not container_selector:
                return []
            
            review_containers = self.driver.find_elements(By.CSS_SELECTOR, container_selector)
            
            for container in review_containers:
                try:
                    review_data = {
                        'product_name': product['title'],
                        'product_category': product['category'],
                        'product_url': product['url'],
                        'product_price': product.get('price', 'N/A'),
                        'scraped_at': datetime.now().isoformat()
                    }
                    
                    # Texte de la review
                    if review_selectors.get('text'):
                        try:
                            text_elem = container.find_element(By.CSS_SELECTOR, review_selectors['text'])
                            review_data['review_text'] = text_elem.text.strip()
                        except:
                            review_data['review_text'] = ''
                    
                    # Titre de la review
                    if review_selectors.get('title'):
                        try:
                            title_elem = container.find_element(By.CSS_SELECTOR, review_selectors['title'])
                            review_data['review_title'] = title_elem.text.strip()
                        except:
                            review_data['review_title'] = ''
                    
                    # Note de la review
                    if review_selectors.get('rating'):
                        try:
                            rating_elem = container.find_element(By.CSS_SELECTOR, review_selectors['rating'])
                            rating_text = rating_elem.get_attribute('textContent') or rating_elem.text
                            # Extraire le chiffre de la note
                            import re
                            rating_match = re.search(r'(\d+(?:\.\d+)?)', rating_text)
                            review_data['user_rating'] = float(rating_match.group(1)) if rating_match else None
                        except:
                            review_data['user_rating'] = None
                    
                    # Auteur
                    if review_selectors.get('author'):
                        try:
                            author_elem = container.find_element(By.CSS_SELECTOR, review_selectors['author'])
                            review_data['reviewer_name'] = author_elem.text.strip()
                        except:
                            review_data['reviewer_name'] = 'Anonymous'
                    
                    # Date
                    if review_selectors.get('date'):
                        try:
                            date_elem = container.find_element(By.CSS_SELECTOR, review_selectors['date'])
                            review_data['review_date'] = date_elem.text.strip()
                        except:
                            review_data['review_date'] = 'N/A'
                    
                    # Ajouter si on a du contenu utile
                    if (review_data.get('review_text') and len(review_data['review_text']) > 10) or \
                       (review_data.get('review_title') and len(review_data['review_title']) > 5):
                        reviews.append(review_data)
                    
                except Exception as e:
                    continue  # Ignorer les reviews problématiques
            
            return reviews
            
        except Exception as e:
            print(f"❌ Erreur extraction reviews: {e}")
            return []
    
    def _go_to_next_page(self):
        """Tente de passer à la page suivante"""
        
        try:
            # Sélecteurs de bouton "suivant"
            next_selectors = [
                '.a-pagination .a-last a',
                'a[aria-label="Next page"]',
                '.a-pagination li:last-child a',
                'a[href*="pageNumber"]'
            ]
            
            for selector in next_selectors:
                try:
                    next_button = self.driver.find_element(By.CSS_SELECTOR, selector)
                    if next_button.is_enabled():
                        next_button.click()
                        time.sleep(2)
                        return True
                except:
                    continue
            
            return False
            
        except Exception as e:
            print(f"⚠️ Erreur navigation page suivante: {e}")
            return False
    
    def _clean_review_data(self, df):
        """Nettoie et optimise les données récupérées"""
        
        try:
            print("🧹 Nettoyage des données...")
            
            # Supprimer les doublons
            initial_count = len(df)
            df = df.drop_duplicates(subset=['review_text', 'product_name'], keep='first')
            print(f"📝 Doublons supprimés: {initial_count - len(df)}")
            
            # Nettoyer les textes
            df['review_text'] = df['review_text'].str.strip()
            df['review_title'] = df['review_title'].str.strip()
            df['product_name'] = df['product_name'].str.strip()
            
            # Filtrer les reviews trop courtes
            df = df[df['review_text'].str.len() > 15]
            
            # Ajouter des métriques
            df['review_length'] = df['review_text'].str.len()
            df['word_count'] = df['review_text'].str.split().str.len()
            
            # Nettoyer les ratings
            df['user_rating'] = pd.to_numeric(df['user_rating'], errors='coerce')
            
            print(f"✅ {len(df)} reviews nettoyées")
            return df
            
        except Exception as e:
            print(f"❌ Erreur nettoyage: {e}")
            return df
    
    def save_data(self, df, filename=None):
        """Sauvegarde les données avec horodatage"""
        
        try:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"../data/raw/reviews_robustes_{timestamp}.csv"
            
            import os
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            df.to_csv(filename, index=False, encoding='utf-8')
            print(f"💾 Données sauvegardées: {filename}")
            
            # Sauvegarder aussi en JSON pour backup
            json_filename = filename.replace('.csv', '.json')
            df.to_json(json_filename, orient='records', force_ascii=False, indent=2)
            
            return filename
            
        except Exception as e:
            print(f"❌ Erreur sauvegarde: {e}")
            return None
    
    def close(self):
        """Ferme proprement le scraper"""
        if self.driver:
            try:
                self.driver.quit()
                print("✅ Driver scraper fermé")
            except:
                pass
            self.driver = None

print("✅ Scraper robuste créé!")