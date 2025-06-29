# ============================================================================
# WORKFLOW PRINCIPAL ROBUSTE
# ============================================================================

def robust_reviews_workflow(category="laptop", site="amazon", max_products=10, reviews_per_rating=50, headless=False):
    """
    Workflow principal ultra-robuste pour scraper les reviews de produits
    
    Phase 1: Détection automatique des balises (Scout)
    Phase 2: Scraping des reviews avec balises validées (Scraper)
    
    Args:
        category: catégorie de produits (ex: "laptop", "smartphone")
        site: site à scraper ("amazon" ou "ebay")
        max_products: nombre max de produits à analyser
        reviews_per_rating: nombre de reviews à récupérer par note
        headless: mode sans interface (True) ou visible (False)
    """
    
    print("="*100)
    print("🚀 WORKFLOW ROBUSTE - SCRAPING REVIEWS DE PRODUITS")
    print("="*100)
    print(f"📦 Catégorie: {category}")
    print(f"🌐 Site: {site}")
    print(f"📊 Produits max: {max_products}")
    print(f"⭐ Reviews par note: {reviews_per_rating}")
    print(f"👁️ Mode: {'Headless' if headless else 'Visible'}")
    print(f"📈 Total estimé: {max_products * 5 * reviews_per_rating} reviews max")
    print()
    
    # Variables pour cleanup
    scout = None
    scraper = None
    
    try:
        # ====================================================================
        # PHASE 1: DÉTECTION DES BALISES (SCOUT)
        # ====================================================================
        print("🔍 PHASE 1: DÉTECTION AUTOMATIQUE DES BALISES")
        print("-" * 70)
        
        scout = RobustProductReviewScout()
        
        # Setup du driver scout
        if not scout.setup_robust_driver(headless=True, timeout=30):
            print("❌ Impossible d'initialiser le scout")
            return None
        
        # Détection des sélecteurs
        site_url = f"https://www.{site}.com"
        detected_selectors = scout.detect_site_selectors(site_url, category)
        
        if not detected_selectors or not detected_selectors.get('products'):
            print("❌ Échec de la détection des sélecteurs")
            scout.close()
            return None
        
        print("✅ Sélecteurs détectés avec succès!")
        print(f"📦 Produits: {list(detected_selectors['products'].keys())}")
        print(f"📝 Reviews: {list(detected_selectors['reviews'].keys())}")
        
        # Fermer le scout
        scout.close()
        scout = None
        
        print("\n" + "="*70)
        
        # ====================================================================
        # PHASE 2: SCRAPING DES REVIEWS (SCRAPER)
        # ====================================================================
        print("📊 PHASE 2: SCRAPING DES REVIEWS")
        print("-" * 70)
        
        scraper = RobustProductReviewScraper()
        scraper.selectors = detected_selectors  # Utiliser les sélecteurs détectés
        
        # Setup du driver scraper
        if not scraper.setup_robust_driver(headless=headless, timeout=60):
            print("❌ Impossible d'initialiser le scraper")
            return None
        
        # Avertissement utilisateur
        if not headless:
            print("\n🚨 AVERTISSEMENT: Scraping sur site réel en cours!")
            print("⏰ Durée estimée: {:.1f} minutes".format(max_products * 3))
            print("📝 Respectez les ToS et les limitations de débit")
            
            response = input("\n🔄 Continuer? (o/n): ").strip().lower()
            if response not in ['o', 'oui', 'y', 'yes', '']:
                print("⏹️ Arrêt demandé par l'utilisateur")
                scraper.close()
                return None
        
        # Lancement du scraping
        print(f"\n🎯 Début du scraping pour '{category}' sur {site}...")
        
        df_reviews = scraper.scrape_category_reviews(
            category=category,
            site=site,
            max_products=max_products,
            reviews_per_rating=reviews_per_rating
        )
        
        # ====================================================================
        # PHASE 3: RÉSULTATS ET SAUVEGARDE
        # ====================================================================
        if df_reviews.empty:
            print("❌ Aucune review récupérée")
            scraper.close()
            return None
        
        print("\n" + "="*70)
        print("📊 ANALYSE DES RÉSULTATS")
        print("-" * 70)
        
        # Statistiques détaillées
        stats = {
            'total_reviews': len(df_reviews),
            'unique_products': df_reviews['product_name'].nunique(),
            'avg_review_length': df_reviews['review_length'].mean(),
            'rating_distribution': df_reviews['user_rating'].value_counts().sort_index(),
            'top_products': df_reviews['product_name'].value_counts().head(5)
        }
        
        print(f"✅ Total reviews: {stats['total_reviews']}")
        print(f"📦 Produits uniques: {stats['unique_products']}")
        print(f"📝 Longueur moyenne: {stats['avg_review_length']:.0f} caractères")
        print(f"⭐ Distribution des notes:")
        for rating, count in stats['rating_distribution'].items():
            if pd.notna(rating):
                print(f"   {rating} étoiles: {count} reviews")
        
        print(f"\n🏆 Top produits par nombre de reviews:")
        for product, count in stats['top_products'].items():
            print(f"   • {product[:50]}... ({count} reviews)")
        
        # Sauvegarde
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"../data/raw/{site}_{category}_{timestamp}.csv"
        
        saved_file = scraper.save_data(df_reviews, filename)
        
        # Aperçu des données
        print(f"\n📋 APERÇU DES DONNÉES (5 premières reviews):")
        print("-" * 70)
        
        sample_cols = ['product_name', 'user_rating', 'review_text', 'reviewer_name']
        available_cols = [col for col in sample_cols if col in df_reviews.columns]
        
        for i, row in df_reviews[available_cols].head(5).iterrows():
            print(f"\nReview {i+1}:")
            for col in available_cols:
                value = str(row[col])
                if col == 'review_text' and len(value) > 100:
                    value = value[:100] + "..."
                elif col == 'product_name' and len(value) > 50:
                    value = value[:50] + "..."
                print(f"  {col}: {value}")
        
        # Fermer le scraper
        scraper.close()
        scraper = None
        
        print("\n" + "="*100)
        print("🎉 WORKFLOW TERMINÉ AVEC SUCCÈS!")
        print("="*100)
        
        return df_reviews
        
    except KeyboardInterrupt:
        print("\n⏹️ Arrêt demandé par l'utilisateur")
        return None
        
    except Exception as e:
        print(f"\n❌ Erreur workflow: {e}")
        import traceback
        traceback.print_exc()
        return None
        
    finally:
        # Cleanup des ressources
        if scout:
            try:
                scout.close()
            except:
                pass
        if scraper:
            try:
                scraper.close()
            except:
                pass

def quick_scout_test(site="amazon", category="laptop"):
    """
    Test rapide du scout uniquement
    """
    print(f"🧪 TEST RAPIDE - Scout pour {category} sur {site}")
    print("-" * 50)
    
    scout = RobustProductReviewScout()
    
    try:
        if scout.setup_robust_driver(headless=True):
            site_url = f"https://www.{site}.com"
            selectors = scout.detect_site_selectors(site_url, category)
            
            if selectors:
                print("✅ Test scout réussi!")
                print(f"📦 Sélecteurs produits: {list(selectors['products'].keys())}")
                print(f"📝 Sélecteurs reviews: {list(selectors['reviews'].keys())}")
                return selectors
            else:
                print("❌ Test scout échoué")
                return None
        else:
            print("❌ Impossible de créer le driver scout")
            return None
            
    except Exception as e:
        print(f"❌ Erreur test scout: {e}")
        return None
        
    finally:
        scout.close()

def robust_reviews_menu():
    """
    Menu principal pour le workflow robuste
    """
    print("="*100)
    print("🎯 WORKFLOW ROBUSTE - REVIEWS DE PRODUITS")
    print("="*100)
    print()
    print("1️⃣ Workflow complet (scout + scraper)")
    print("2️⃣ Test scout uniquement")
    print("3️⃣ Configuration avancée")
    print("4️⃣ Voir fichiers de données")
    print("5️⃣ Quitter")
    print()
    
    while True:
        try:
            choice = input("👉 Votre choix (1-5): ").strip()
            
            if choice == '1':
                print("\n📋 CONFIGURATION DU WORKFLOW COMPLET")
                print("-" * 50)
                
                category = input("🏷️ Catégorie (ex: laptop, smartphone): ").strip() or "laptop"
                site = input("🌐 Site (amazon/ebay): ").strip() or "amazon"
                
                try:
                    max_products = int(input("📦 Nombre de produits (défaut: 5): ") or "5")
                    reviews_per_rating = int(input("⭐ Reviews par note (défaut: 20): ") or "20")
                except ValueError:
                    max_products, reviews_per_rating = 5, 20
                
                headless_choice = input("👁️ Mode headless? (o/n, défaut: n): ").strip().lower()
                headless = headless_choice in ['o', 'oui', 'y', 'yes']
                
                print(f"\n🚀 Lancement du workflow...")
                result = robust_reviews_workflow(category, site, max_products, reviews_per_rating, headless)
                
                if result is not None:
                    print(f"\n✅ Workflow terminé - {len(result)} reviews récupérées")
                else:
                    print("\n❌ Workflow échoué")
                
                return result
                
            elif choice == '2':
                print("\n🧪 TEST SCOUT")
                print("-" * 30)
                
                site = input("🌐 Site (amazon/ebay): ").strip() or "amazon"
                category = input("🏷️ Catégorie: ").strip() or "laptop"
                
                result = quick_scout_test(site, category)
                if result:
                    print("✅ Scout fonctionne correctement!")
                else:
                    print("❌ Problème avec le scout")
                
            elif choice == '3':
                print("\n⚙️ CONFIGURATION AVANCÉE")
                print("-" * 40)
                print("📖 Paramètres disponibles dans robust_reviews_workflow():")
                print("   • category: catégorie de produits")
                print("   • site: amazon ou ebay")
                print("   • max_products: nombre de produits max")
                print("   • reviews_per_rating: reviews par note (1-5)")
                print("   • headless: mode sans interface")
                print("\n💡 Exemple:")
                print("   df = robust_reviews_workflow('gaming laptop', 'amazon', 8, 30, False)")
                
            elif choice == '4':
                print("\n📁 FICHIERS DE DONNÉES")
                print("-" * 30)
                
                import os
                data_dir = "../data/raw"
                
                if os.path.exists(data_dir):
                    files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
                    if files:
                        print("📄 Fichiers CSV trouvés:")
                        for f in sorted(files, reverse=True)[:10]:  # 10 plus récents
                            size = os.path.getsize(os.path.join(data_dir, f)) / 1024  # KB
                            print(f"   • {f} ({size:.1f} KB)")
                    else:
                        print("❌ Aucun fichier CSV trouvé")
                else:
                    print("❌ Dossier data/raw non trouvé")
                
            elif choice == '5':
                print("👋 À bientôt!")
                break
                
            else:
                print("❌ Choix invalide")
                
        except KeyboardInterrupt:
            print("\n👋 À bientôt!")
            break
        except Exception as e:
            print(f"❌ Erreur: {e}")
    
    return None

# Configuration d'exemples prêts à l'emploi
ROBUST_EXAMPLES = {
    'laptops_gaming': {
        'category': 'gaming laptop',
        'site': 'amazon',
        'max_products': 8,
        'reviews_per_rating': 25,
        'headless': False
    },
    'smartphones': {
        'category': 'smartphone',
        'site': 'amazon', 
        'max_products': 6,
        'reviews_per_rating': 30,
        'headless': False
    },
    'headphones': {
        'category': 'wireless headphones',
        'site': 'amazon',
        'max_products': 10,
        'reviews_per_rating': 20,
        'headless': False
    }
}

def run_example(example_name):
    """Exécute un exemple prédéfini"""
    if example_name in ROBUST_EXAMPLES:
        config = ROBUST_EXAMPLES[example_name]
        print(f"🚀 Lancement exemple: {example_name}")
        return robust_reviews_workflow(**config)
    else:
        print(f"❌ Exemple '{example_name}' non trouvé")
        print(f"📋 Disponibles: {list(ROBUST_EXAMPLES.keys())}")
        return None

print("✅ Workflow robuste prêt!")
print("📖 Utilisez robust_reviews_menu() pour commencer")
print("🚀 Ou run_example('laptops_gaming') pour un test rapide")