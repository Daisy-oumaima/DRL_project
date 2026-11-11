# DRL Trading avec VAE-LSTM : PPO / SAC / TD3 sur le SPY

Système de trading automatisé qui combine **apprentissage par renforcement profond (DRL)** et **Variational Autoencoder (VAE-LSTM)**.
Idée centrale : compresser 20 jours de données de marché (24 à 26 variables) en un vecteur latent de 32 dimensions, puis donner ce vecteur comme *état* à l'agent, au lieu des variables brutes et bruitées.

> Projet académique (ENSAM Casablanca, module Deep Reinforcement Learning), réalisé en binôme avec **Imane Afandi**, encadré par **Pr. Badr Hirchoua**.
> ⚠️ Projet pédagogique : ce n'est pas un conseil financier.

## Pipeline

1. **Données** (2019–2024, ~1 250 jours) : SPY (yfinance), indice VIX, taux directeur Fed Funds (FRED, repris en fréquence journalière).
2. **Features** : rendements (1/5/20 j), moyennes mobiles, MACD, RSI, Bandes de Bollinger, ATR, volume, VIX et taux Fed dérivés.
3. **VAE-LSTM** : encodeur LSTM à 2 couches, fenêtre de 20 jours, espace latent de 32 dimensions, perte ELBO (MSE + 0,5 × KL). Visualisation de l'espace latent par t-SNE.
4. **Environnement Gymnasium** personnalisé : état = vecteur latent + position + ratio de capital (34 dimensions) ; récompense basée sur un Sharpe glissant ; coûts de transaction de 0,1 % et glissement (slippage) de 0,05 % ; capital initial de 10 000 $.
5. **Agents** (Stable-Baselines3) : PPO, SAC, TD3, 100 000 pas chacun, sélection sur la validation.
6. **Évaluation hors échantillon** (177 jours, 19/04/2023 → 29/12/2023) avec QuantStats, plus un tableau de bord Streamlit.

Découpage temporel : 70 % entraînement / 15 % validation / 15 % test, normalisation ajustée sur le train uniquement.

## Résultats (jeu de test, 177 jours)

| Métrique | **DRL + VAE (SAC)** | DRL sans VAE (SAC) | Buy & Hold |
|---|---|---|---|
| Rendement total | **+19,38 %** | +4,80 % | +16,58 % |
| Ratio de Sharpe | **2,594** | 0,775 | 1,915 |
| Ratio de Sortino | 3,789 | 1,012 | 3,172 |
| Drawdown maximum | **−6,63 %** | −8,19 % | −9,97 % |
| Nombre de trades | 127 | 143 | 1 |

Comparaison des algorithmes (meilleur rendement en validation) : **SAC +8,76 %** > TD3 +5,82 % > PPO +1,84 %.

### Lecture honnête des résultats
- Un seul actif (SPY), une seule période de test courte (177 jours) : **pas de preuve de robustesse**.
- Résultats obtenus avec une seule exécution (à compléter avec plusieurs graines aléatoires et des intervalles de confiance).
- L'écart entre la perte d'entraînement (~0,51) et de validation (~1,71) du VAE montre la difficulté de généraliser à de nouveaux régimes de marché.

## Contenu du dépôt

| Fichier | Description |
|---|---|
| `notebooks/01_drl_vae_sac_v1_rapport.ipynb` | Pipeline complet de la version du rapport, avec les sorties |
| `notebooks/02_drl_vae_lstm_v2_experimental.ipynb` | Version 2 expérimentale : Fear & Greed, Put/Call, VAE-LSTM sur 30 jours, actions continues, récompense Sharpe + pénalité de drawdown, PPO/SAC/TD3 réimplémentés en PyTorch. Sorties non incluses |

## Lancer le projet

Conçu pour **Google Colab** (ou Python 3.10+) :

```bash
pip install -r requirements.txt
jupyter notebook notebooks/01_drl_vae_sac_v1_rapport.ipynb
```

Le tableau de bord Streamlit du notebook 01 utilise `pyngrok` : renseigne ton propre jeton ngrok dans la variable `NGROK_TOKEN` (ne le publie jamais).

## Pistes d'amélioration
- Validation *walk-forward* sur plusieurs périodes et plusieurs actifs
- Plusieurs graines et intervalles de confiance
- Comparaison avec d'autres encodeurs (PCA, autoencodeur simple) pour isoler l'apport du VAE

## Auteures
Oumaima Kanas, Imane Afandi · ENSAM Casablanca (IA & Génie Informatique)
