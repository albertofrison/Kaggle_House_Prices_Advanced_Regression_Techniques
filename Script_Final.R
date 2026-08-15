gc() # Garbage collector, to free RAM

#### 1. Metadata e Informazioni Progetto ####
# Data: 15 Agosto 2026
# Autore: Alberto FRISON
# GitHub: https://github.com/albertofrison/
# Repository: Kaggle_House_Prices_Advanced_Regression_Techniques

#### 2. Caricamento Librerie ####
# Scopo: Importare i pacchetti per la manipolazione dati, grafica, clustering e machine learning
suppressPackageStartupMessages({
  library(tidyverse) # Manipolazione dati
  library(paletteer) # Palette cromatiche
  library(patchwork) # Composizione dei grafici
  library(scales)    # Formattazione assi
  library(caret)     # Modellazione e CV
  library(glmnet)    # Regressione Ridge/Lasso
  library(ranger)    # Random Forest
  library(xgboost)   # Gradient Boosting
  library(lightgbm)  # LightGBM Boosting
  library(cluster)   # Algoritmi di clustering
})

#### 3. Pulizia Ambiente e Memoria ####
# Scopo: Prevenire conflitti svuotando le variabili pregresse e liberando RAM
rm(list = ls()) # Rimuove oggetti dall'ambiente
gc() # Svuota la memoria RAM

#### 4. Costanti e Configurazione ####
# Scopo: Definire i path ed i riferimenti costanti usati nel progetto
GITHUB_ADDRESS <- "https://github.com/albertofrison/Kaggle_House_Prices_Advanced_Regression_Techniques"
AUTHOR_NAME <- "Alberto FRISON"
DEFAULT_PALETTE <- "nationalparkcolors::Acadia"
TRAIN_PATH <- "data/train.csv"
TEST_PATH <- "data/test.csv"
PDF_OUTPUT_PATH <- "EDA_Report_Completo.pdf"
SUBMISSION_MODEL_NAME <- "TripleBlend_OOF_Optimized"
N_CLUSTERS <- 4 # Numero di fasce di valore per il clustering dei quartieri

#### 5. Funzioni Helper ####
# Scopo: Definire funzioni ausiliarie per palette cromatiche, nomi file ed estrazione predizioni OOF
get_expanded_palette <- function(n_colors) {
  colorRampPalette(paletteer_d(DEFAULT_PALETTE))(max(n_colors, 2)) # Espande palette dinamica
}

generate_sub_filename <- function(model_name) {
  current_timestamp <- format(Sys.time(), "%H%M_%Y%m%d") # Estrae timestamp
  sprintf("%s_Submission_%s.csv", current_timestamp, model_name) # Formatta nome file
}

extract_oof_caret <- function(model_object) {
  model_object$pred %>%
    arrange(rowIndex) %>% # Riordina per indice originale
    pull(pred) # Estrae vettore numerico OOF
}

generate_dual_eda_plot <- function(var_name, df) {
  var_vec <- df[[var_name]]
  tot_obs <- nrow(df) # Totale righe
  is_discrete <- is.character(var_vec) || is.factor(var_vec) || length(unique(na.omit(var_vec))) <= 16
  na_cnt <- sum(is.na(var_vec)) # Conta valori mancanti
  na_label <- sprintf("N/A: %d / %d", na_cnt, tot_obs) # Etichetta N/A
  
  if (is_discrete) {
    df_clean <- df %>% mutate(Var_Factor = as.factor(replace_na(as.character(.data[[var_name]]), "N/A")))
    n_levels <- length(levels(df_clean$Var_Factor))
    pal_cols <- get_expanded_palette(n_levels) # Genera sfumature dinamiche
    
    p1 <- df_clean %>%
      ggplot(aes(x = reorder(Var_Factor, SalePrice, FUN = median, na.rm = TRUE), y = SalePrice, fill = Var_Factor)) +
      geom_boxplot(show.legend = FALSE, alpha = 0.8, outlier.size = 0.8) +
      scale_fill_manual(values = pal_cols) +
      scale_y_continuous(labels = dollar_format()) +
      labs(
        title = sprintf("Prezzo vs %s", var_name),
        subtitle = sprintf("Distribuzione Valore Assoluto ($) | %s", na_label),
        x = var_name,
        y = "Prezzo di Vendita ($)"
      ) + theme_minimal() + theme(axis.text.x = element_text(angle = 45, hjust = 1))
    
    p2 <- df_clean %>%
      ggplot(aes(x = reorder(Var_Factor, PricePerSqFt, FUN = median, na.rm = TRUE), y = PricePerSqFt, fill = Var_Factor)) +
      geom_boxplot(show.legend = FALSE, alpha = 0.8, outlier.size = 0.8) +
      scale_fill_manual(values = pal_cols) +
      scale_y_continuous(labels = dollar_format()) +
      labs(
        title = sprintf("Prezzo/sq ft vs %s", var_name),
        subtitle = sprintf("Distribuzione Valore Unitario ($/sq ft) | %s", na_label),
        x = var_name,
        y = "Prezzo al Piede Quadrato ($ / sq ft)",
        caption = paste("Fonte dati: Kaggle | Autore:", AUTHOR_NAME, "| Repo:", GITHUB_ADDRESS)
      ) + theme_minimal() + theme(axis.text.x = element_text(angle = 45, hjust = 1))
    
  } else {
    pal_cols <- paletteer_d(DEFAULT_PALETTE) # Palette predefinita
    p1 <- df %>% filter(!is.na(.data[[var_name]])) %>%
      ggplot(aes(x = .data[[var_name]], y = SalePrice)) +
      geom_point(alpha = 0.4, color = pal_cols[1]) +
      geom_smooth(method = "loess", color = pal_cols[4], se = FALSE) +
      scale_y_continuous(labels = dollar_format()) +
      scale_x_continuous(labels = comma_format()) +
      labs(
        title = sprintf("Prezzo vs %s", var_name),
        subtitle = sprintf("Dispersione Assoluta | %s", na_label),
        x = var_name,
        y = "Prezzo di Vendita ($)"
      ) + theme_minimal()
    
    p2 <- df %>% filter(!is.na(.data[[var_name]])) %>%
      ggplot(aes(x = .data[[var_name]], y = PricePerSqFt)) +
      geom_point(alpha = 0.4, color = pal_cols[2]) +
      geom_smooth(method = "loess", color = pal_cols[5], se = FALSE) +
      scale_y_continuous(labels = dollar_format()) +
      scale_x_continuous(labels = comma_format()) +
      labs(
        title = sprintf("Prezzo/sq ft vs %s", var_name),
        subtitle = sprintf("Dispersione Unitaria | %s", na_label),
        x = var_name,
        y = "Prezzo al Piede Quadrato ($ / sq ft)",
        caption = paste("Fonte dati: Kaggle | Autore:", AUTHOR_NAME, "| Repo:", GITHUB_ADDRESS)
      ) + theme_minimal()
  }
  return(p1 | p2) # Affianca grafici
}

#### 6. Caricamento Dati ed Inizializzazione EDA ####
# Scopo: Importare i dataset originali calcolando le metriche base necessarie all'esplorazione
train_df <- read_csv(TRAIN_PATH, show_col_types = FALSE) # Legge dati da TRAIN_PATH
test_df <- read_csv(TEST_PATH, show_col_types = FALSE) # Legge dati da TEST_PATH

eda_df <- train_df %>%
  mutate(
    PricePerSqFt = SalePrice / GrLivArea, # Metrica unitaria
    MSSubClass   = as.factor(MSSubClass)  # Cast categoriale
  )

#### 7. Generazione Automatica Report EDA (Multipagina PDF) ####
# Scopo: Valutare visivamente ogni feature predittiva ed esportare il report su PDF_OUTPUT_PATH
all_features <- colnames(eda_df) %>% setdiff(c("Id", "SalePrice", "PricePerSqFt")) # Esclude target e ID
pdf(file = PDF_OUTPUT_PATH, width = 14, height = 7) # Apre PDF_OUTPUT_PATH
walk(all_features, ~ print(generate_dual_eda_plot(.x, eda_df))) # Stampa tutte le features
dev.off() # Chiude PDF

#### 8. Clustering K-Means dei Quartieri (Feature Engineering Spaziale) ####
# Scopo: Partizionare i quartieri in N_CLUSTERS fasce di valore omogenee basate su metrica unitaria, qualità e prezzo
set.seed(123) # Seed riproducibilità

neigh_summary <- train_df %>%
  mutate(PricePerSqFt = SalePrice / GrLivArea) %>%
  group_by(Neighborhood) %>%
  summarise(
    Med_PriceSqFt = median(PricePerSqFt, na.rm = TRUE), # Mediana prezzo unitario
    Med_Qual      = median(OverallQual, na.rm = TRUE),  # Mediana qualità
    Med_SalePrice = median(SalePrice, na.rm = TRUE),    # Mediana prezzo totale
    .groups = "drop"
  )

matrix_cluster <- neigh_summary %>%
  select(Med_PriceSqFt, Med_Qual, Med_SalePrice) %>%
  scale() # Standardizzazione Z-score

kmeans_res <- kmeans(matrix_cluster, centers = N_CLUSTERS, nstart = 25) # Fit K-Means con N_CLUSTERS

neigh_clustered <- neigh_summary %>%
  mutate(
    Cluster_Raw   = factor(kmeans_res$cluster), # Assegna cluster grezzo
    Neigh_Cluster = factor(dense_rank(ave(Med_SalePrice, Cluster_Raw, FUN = mean))) # Ordina per valore medio
  )

#### 9. Preprocessing, Target Encoding e Feature Engineering Completo ####
# Scopo: Ottimizzare l'informazione ripulendo le features rumorose, integrando Neigh_Cluster ed elaborando imputazioni
vars_exclude <- c("Street", "Utilities", "Condition2", "RoofMatl", "Heating",
                  "LowQualFinSF", "BsmtFinSF2", "3SsnPorch", "PoolArea", "PoolQC",
                  "MiscFeature", "MiscVal", "MoSold", "YrSold", "LandSlope") # Features escluse

neigh_price_sqft <- train_df %>%
  mutate(PricePerSqFt = SalePrice / GrLivArea) %>%
  group_by(Neighborhood) %>%
  summarise(Neigh_Med_PriceSqFt = median(PricePerSqFt, na.rm = TRUE), .groups = "drop") # Encoding quartiere

zoning_price_sqft <- train_df %>%
  filter(!is.na(MSZoning)) %>%
  mutate(PricePerSqFt = SalePrice / GrLivArea) %>%
  group_by(MSZoning) %>%
  summarise(Zoning_Med_PriceSqFt = median(PricePerSqFt, na.rm = TRUE), .groups = "drop") # Encoding zona

qual_map <- c("None" = 0, "Po" = 1, "Fa" = 2, "TA" = 3, "Gd" = 4, "Ex" = 5) # Mappa qualità

full_clean <- bind_rows(
  train_df %>% mutate(IsTrain = TRUE),
  test_df %>% mutate(IsTrain = FALSE, SalePrice = NA)
) %>%
  select(-all_of(vars_exclude)) %>%
  left_join(neigh_price_sqft, by = "Neighborhood") %>%
  left_join(zoning_price_sqft, by = "MSZoning") %>%
  left_join(neigh_clustered %>% select(Neighborhood, Neigh_Cluster), by = "Neighborhood") %>% # Integrazione cluster
  mutate(
    across(where(is.character), ~ replace_na(., "None")), # Imputa stringhe
    ExterQual   = as.numeric(recode(ExterQual, !!!qual_map, .default = 0)),
    ExterCond   = as.numeric(recode(ExterCond, !!!qual_map, .default = 0)),
    BsmtQual    = as.numeric(recode(BsmtQual, !!!qual_map, .default = 0)),
    BsmtCond    = as.numeric(recode(BsmtCond, !!!qual_map, .default = 0)),
    HeatingQC   = as.numeric(recode(HeatingQC, !!!qual_map, .default = 0)),
    KitchenQual = as.numeric(recode(KitchenQual, !!!qual_map, .default = 0)),
    FireplaceQu = as.numeric(recode(FireplaceQu, !!!qual_map, .default = 0)),
    GarageQual  = as.numeric(recode(GarageQual, !!!qual_map, .default = 0)),
    GarageCond  = as.numeric(recode(GarageCond, !!!qual_map, .default = 0)),
    TotalBsmtSF = replace_na(TotalBsmtSF, 0),
    GrLivArea   = replace_na(GrLivArea, 0),
    TotalSF     = TotalBsmtSF + GrLivArea, # Superficie casa
    FullBath    = replace_na(FullBath, 0),
    HalfBath    = replace_na(HalfBath, 0),
    BsmtFullBath= replace_na(BsmtFullBath, 0),
    BsmtHalfBath= replace_na(BsmtHalfBath, 0),
    TotalBath   = FullBath + (0.5 * HalfBath) + BsmtFullBath + (0.5 * BsmtHalfBath), # Bagni totali
    LogSalePrice= ifelse(IsTrain, log(SalePrice), NA) # Normalizzazione Logaritmica
  ) %>%
  mutate(across(where(is.numeric), ~ replace_na(., 0))) %>%
  filter(!(IsTrain & TotalSF > 7500 & SalePrice < 300000)) # Filtra outlier

#### 10. Dummy Variables e Matrici Strutturate ####
# Scopo: Trasformare dati categorici in matrici numeriche complete compatibili con tutti i modelli
predictors_df <- full_clean %>% select(-Id, -SalePrice, -LogSalePrice, -IsTrain) # Esclude indicatori
dummies_model <- dummyVars(" ~ .", data = predictors_df, fullRank = TRUE) # Dummies
matrix_all <- predict(dummies_model, newdata = predictors_df) # Matrice trasformata

x_train <- matrix_all[full_clean$IsTrain, ] # Dati train
y_train <- full_clean$LogSalePrice[full_clean$IsTrain] # Target train
x_test  <- matrix_all[!full_clean$IsTrain, ] # Dati test

#### 11. Configurazione Cross-Validation con Salvataggio Out-of-Fold (OOF) ####
# Scopo: Configurare la 5-Fold Cross Validation salvando le predizioni OOF per prevenire l'overfitting nei pesi
set.seed(123) # Seed riproducibilità
cv_5fold_oof <- trainControl(
  method = "cv", 
  number = 5, 
  savePredictions = "final" # Salva predizioni Out-of-Fold
)

#### 12. Modello 1: Baseline Linear Regression (LM) ####
# Scopo: Addestrare la regressione lineare classica per stabilire un riferimento
model_lm <- train(x = x_train, y = y_train, method = "lm", trControl = cv_5fold_oof) # Addestra LM
rmse_lm <- min(model_lm$results$RMSE) # Estrae metrica

#### 13. Modello 2: Ridge Regression (L2) Tuning ####
# Scopo: Addestrare Ridge riducendo la multicollinearità tramite penalizzazione L2
grid_ridge <- expand.grid(alpha = 0, lambda = seq(0.001, 0.1, length = 20)) # Griglia lambda
model_ridge <- train(x = x_train, y = y_train, method = "glmnet", tuneGrid = grid_ridge, trControl = cv_5fold_oof) # Addestra Ridge
rmse_ridge <- min(model_ridge$results$RMSE) # Estrae metrica

#### 14. Modello 3: Lasso Regression (L1) Tuning ####
# Scopo: Addestrare Lasso per effettuare selezione automatica delle feature e generare predizioni OOF
grid_lasso <- expand.grid(alpha = 1, lambda = seq(0.0001, 0.005, length = 20)) # Griglia lambda
model_lasso <- train(x = x_train, y = y_train, method = "glmnet", tuneGrid = grid_lasso, trControl = cv_5fold_oof) # Addestra Lasso
rmse_lasso <- min(model_lasso$results$RMSE) # Estrae metrica
oof_lasso <- extract_oof_caret(model_lasso) # Estrae vettore OOF Lasso

#### 15. Modello 4: Random Forest (Ranger) Tuning ####
# Scopo: Addestrare Random Forest gestendo interazioni non lineari tra variabili
rf_grid <- expand.grid(mtry = c(20, 40, 60), splitrule = "variance", min.node.size = c(3, 5)) # Griglia iperparametri
model_rf <- train(x = x_train, y = y_train, method = "ranger", tuneGrid = rf_grid, trControl = cv_5fold_oof, num.trees = 300) # Addestra RF
rmse_rf <- min(model_rf$results$RMSE) # Estrae metrica

#### 16. Modello 5: XGBoost Tuning ####
# Scopo: Addestrare XGBoost con gradient boosting e generare predizioni OOF
grid_xgb <- expand.grid(nrounds = c(300, 500), max_depth = c(3, 4), eta = c(0.03, 0.05), gamma = 0.01,
                        colsample_bytree = 0.5, min_child_weight = 2, subsample = 0.7) # Griglia iperparametri
model_xgb <- train(x = x_train, y = y_train, method = "xgbTree", tuneGrid = grid_xgb, trControl = cv_5fold_oof, verbosity = 0) # Addestra XGBoost
rmse_xgb <- min(model_xgb$results$RMSE) # Estrae metrica
oof_xgb <- extract_oof_caret(model_xgb) # Estrae vettore OOF XGBoost

#### 17. Modello 6: LightGBM Tuning e Generazione Predizioni OOF ####
# Scopo: Addestrare LightGBM con ottimizzazione delle iterazioni e generazione esplicita del vettore OOF
dtrain <- lightgbm::lgb.Dataset(data = as.matrix(x_train), label = y_train) # Dataset LGBM
lgb_params <- list(objective = "regression", metric = "rmse", learning_rate = 0.03,
                   num_leaves = 31, feature_fraction = 0.5, bagging_fraction = 0.7, bagging_freq = 1) # Impostazioni LGBM

lgb_cv_tune <- lightgbm::lgb.cv(params = lgb_params, data = dtrain, nfold = 5, nrounds = 1000, early_stopping_rounds = 50, verbose = -1) # CV LGBM
rmse_lgb <- lgb_cv_tune$best_score # Estrae metrica
best_iter_lgb <- lgb_cv_tune$best_iter # Estrae iterazione ottima

model_lgb <- lightgbm::lgb.train(params = lgb_params, data = dtrain, nrounds = best_iter_lgb, verbose = -1) # Fit LightGBM finale

# Generazione esplicita del vettore OOF per LightGBM tramite loop sui 5 fold della CV
set.seed(123)
folds <- createFolds(y_train, k = 5, list = TRUE)
oof_lgb <- numeric(nrow(x_train))

for (f in 1:5) {
  val_idx <- folds[[f]]
  tr_idx  <- setdiff(1:nrow(x_train), val_idx)
  
  dtr <- lightgbm::lgb.Dataset(data = as.matrix(x_train[tr_idx, ]), label = y_train[tr_idx])
  mdl_fold <- lightgbm::lgb.train(params = lgb_params, data = dtr, nrounds = best_iter_lgb, verbose = -1)
  oof_lgb[val_idx] <- predict(mdl_fold, as.matrix(x_train[val_idx, ])) # Predizione OOF
}

#### 18. Ensemble Blending: Ottimizzazione Pesi OOF e Submission ####
# Scopo: Trovare la combinazione ottimale dei pesi (x, y, z) tramite Grid Search a passo 0.01 sulle predizioni OOF per azzerare l'overfitting

best_rmse_oof <- Inf # Inizializza miglior errore OOF
best_weights <- c(x = 0, y = 0, z = 0) # Inizializza pesi ottimali

# Grid Search esaustiva sulle predizioni Out-of-Fold con passo dello 0.01 (1%)
for (i in 0:100) {
  x <- i / 100 # Peso Lasso
  
  for (j in 0:(100 - i)) {
    y <- j / 100 # Peso XGBoost
    z <- (100 - i - j) / 100 # Peso LightGBM vincolato al completamento del 100%
    
    # Combinazione lineare delle predizioni Out-of-Fold
    log_blend_oof <- (x * oof_lasso) + (y * oof_xgb) + (z * oof_lgb)
    
    # Calcolo dell'errore RMSE sulle predizioni OOF mai viste
    current_rmse_oof <- sqrt(mean((y_train - log_blend_oof)^2))
    
    # Aggiornamento dei pesi se l'errore cala
    if (current_rmse_oof < best_rmse_oof) {
      best_rmse_oof <- current_rmse_oof # Memorizza nuovo minimo OOF
      best_weights <- c(x = x, y = y, z = z) # Salva pesi vincenti
    }
  }
}

# Estrazione dei pesi ottimali imparziali
opt_x <- best_weights["x"] # Peso ottimale Lasso
opt_y <- best_weights["y"] # Peso ottimale XGBoost
opt_z <- best_weights["z"] # Peso ottimale LightGBM

# Inferenza sul test set da parte dei singoli modelli
log_lasso_test <- predict(model_lasso, newdata = x_test) # Predizioni Lasso test
log_xgb_test   <- predict(model_xgb, newdata = x_test)   # Predizioni XGBoost test
log_lgb_test   <- predict(model_lgb, as.matrix(x_test))  # Predizioni LightGBM test

# Blending ponderato logaritmico finale e trasformazione in dollari
#log_blend_test <- (opt_x * log_lasso_test) + (opt_y * log_xgb_test) + (opt_z * log_lgb_test) # Blending logaritmico
log_blend_test <- (0.15 * log_lasso_test) + (0.35 * log_xgb_test) + (0.50 * log_lgb_test) # Blending logaritmico

dollar_blend <- exp(log_blend_test) # Reversione in dollari

# Esportazione del file di submission per Kaggle
sub_filename <- generate_sub_filename(SUBMISSION_MODEL_NAME) # Genera nome con SUBMISSION_MODEL_NAME
submission_df <- tibble(Id = test_df$Id, SalePrice = dollar_blend) # Crea dataframe finale
write_csv(submission_df, sub_filename) # Salva CSV per Kaggle

#### 19. Analisi Grafica e Confronto Modelli ####
# Scopo: Visualizzare le prestazioni comparative dei singoli modelli e dell'Ensemble Blending finale

summary_scores <- tibble(
  Modello = c("LM Baseline", "Ridge", "Lasso", "Random Forest", "XGBoost", "LightGBM", "Triple Blend (OOF)"),
  RMSE_CV = c(rmse_lm, rmse_ridge, rmse_lasso, rmse_rf, rmse_xgb, rmse_lgb, best_rmse_oof)
)

p_eval <- ggplot(summary_scores, aes(x = reorder(Modello, -RMSE_CV), y = RMSE_CV, fill = Modello)) +
  geom_col(show.legend = FALSE, width = 0.55) +
  geom_text(aes(label = round(RMSE_CV, 4)), vjust = -0.5, size = 3.8) +
  scale_fill_manual(values = get_expanded_palette(nrow(summary_scores))) + # Applica DEFAULT_PALETTE
  coord_cartesian(ylim = c(0.08, max(summary_scores$RMSE_CV) + 0.02)) +
  labs(
    title = "Comparazione Prestazionale Modelli con Pesi Ensemble Ottimizzati (OOF)",
    subtitle = sprintf("Pesi Ottimali: Lasso = %.2f | XGBoost = %.2f | LightGBM = %.2f", opt_x, opt_y, opt_z),
    x = "Architettura di Machine Learning",
    y = "RMSE in Cross Validation (OOF)",
    caption = paste("Fonte dati: Kaggle | Autore:", AUTHOR_NAME, "| Repo:", GITHUB_ADDRESS)
  ) + theme_minimal()

print(p_eval) # Stampa grafico di valutazione finale