gc() # Garbage collector, to free RAM

#### 1. Metadata e Informazioni Progetto ####
# Data: 14 Agosto 2026
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
# Scopo: Prevenire conflitti svuotando le variabili pregresse
rm(list = ls()) # Rimuove oggetti
gc() # Svuota RAM

#### 4. Costanti e Configurazione ####
# Scopo: Definire i path e i riferimenti costanti usati nel progetto
GITHUB_ADDRESS <- "https://github.com/albertofrison/Kaggle_House_Prices_Advanced_Regression_Techniques"
AUTHOR_NAME <- "Alberto FRISON"
DEFAULT_PALETTE <- "nationalparkcolors::Acadia"
TRAIN_PATH <- "data/train.csv"
TEST_PATH <- "data/test.csv"
PDF_OUTPUT_PATH <- "EDA_Report_Completo.pdf"
SUBMISSION_MODEL_NAME <- "TripleBlend_Clustered_Tuned"
N_CLUSTERS <- 4 # Numero di fasce di valore per il clustering dei quartieri

#### 5. Funzioni Helper ####
# Scopo: Evitare la ripetizione di codice per i grafici, i colori e i nomi di file
get_expanded_palette <- function(n_colors) {
  colorRampPalette(paletteer_d(DEFAULT_PALETTE))(max(n_colors, 2)) # Espande sfumature cromatiche
}

generate_sub_filename <- function(model_name) {
  current_timestamp <- format(Sys.time(), "%H%M_%Y%m%d") # Estrae timestamp
  sprintf("%s_Submission_%s.csv", current_timestamp, model_name) # Formatta file
}

generate_dual_eda_plot <- function(var_name, df) {
  var_vec <- df[[var_name]]
  tot_obs <- nrow(df) # Totale righe
  is_discrete <- is.character(var_vec) || is.factor(var_vec) || length(unique(na.omit(var_vec))) <= 16
  na_cnt <- sum(is.na(var_vec)) # Conta mancanti
  na_label <- sprintf("N/A: %d / %d", na_cnt, tot_obs) # Etichetta N/A
  
  if (is_discrete) {
    df_clean <- df %>% mutate(Var_Factor = as.factor(replace_na(as.character(.data[[var_name]]), "N/A")))
    n_levels <- length(levels(df_clean$Var_Factor))
    pal_cols <- get_expanded_palette(n_levels) # Genera palette
    
    p1 <- df_clean %>%
      ggplot(aes(x = reorder(Var_Factor, SalePrice, FUN = median, na.rm = TRUE), y = SalePrice, fill = Var_Factor)) +
      geom_boxplot(show.legend = FALSE, alpha = 0.8, outlier.size = 0.8) +
      scale_fill_manual(values = pal_cols) +
      scale_y_continuous(labels = dollar_format()) +
      labs(
        title = sprintf("Prezzo vs %s", var_name),
        subtitle = sprintf("Distribuzione Valore Assoluto ($) | %s", na_label),
        x = var_name,
        y = "Prezzo ($)"
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
        y = "Prezzo al sq ft",
        caption = paste("Fonte dati: Kaggle | Repo:", GITHUB_ADDRESS)
      ) + theme_minimal() + theme(axis.text.x = element_text(angle = 45, hjust = 1))
    
  } else {
    pal_cols <- paletteer_d(DEFAULT_PALETTE) # Palette base
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
        y = "Prezzo ($)"
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
        y = "Prezzo al sq ft",
        caption = paste("Fonte dati: Kaggle | Repo:", GITHUB_ADDRESS)
      ) + theme_minimal()
  }
  return(p1 | p2) # Affianca grafici
}

#### 6. Caricamento Dati ed Inizializzazione EDA ####
# Scopo: Importare i dataset originali calcolando le metriche base necessarie all'esplorazione
train_df <- read_csv(TRAIN_PATH, show_col_types = FALSE) # Legge TRAIN_PATH
test_df <- read_csv(TEST_PATH, show_col_types = FALSE) # Legge TEST_PATH

eda_df <- train_df %>%
  mutate(
    PricePerSqFt = SalePrice / GrLivArea, # Metrica unitaria
    MSSubClass   = as.factor(MSSubClass)  # Cast categoriale
  )

#### 7. Generazione Automatica Report EDA (Multipagina PDF) ####
# Scopo: Valutare visivamente il comportamento di ogni singola feature predittiva ed esportare su PDF_OUTPUT_PATH
all_features <- colnames(eda_df) %>% setdiff(c("Id", "SalePrice", "PricePerSqFt")) # Esclude var target

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
names(full_clean)

#### 10. Dummy Variables e Matrici Strutturate ####
# Scopo: Trasformare dati categorici in matrici numeriche complete compatibili con tutti i modelli
predictors_df <- full_clean %>% select(-Id, -SalePrice, -LogSalePrice, -IsTrain) # Esclude indicatori
dummies_model <- dummyVars(" ~ .", data = predictors_df, fullRank = TRUE) # Dummies
matrix_all <- predict(dummies_model, newdata = predictors_df) # Matrice trasformata

x_train <- matrix_all[full_clean$IsTrain, ] # Dati train
y_train <- full_clean$LogSalePrice[full_clean$IsTrain] # Target train
x_test  <- matrix_all[!full_clean$IsTrain, ] # Dati test

#### 11. Configurazione Generica Cross-Validation ####
# Scopo: Assicurare consistenza validativa per tutte le ottimizzazioni degli iperparametri
set.seed(123) # Seed riproducibilità
cv_5fold <- trainControl(method = "cv", number = 5) # Setup CV

#### 12. Modello 1: Baseline Linear Regression (LM) ####
# Scopo: Creare una base di paragone iniziale con regressione lineare classica
model_lm <- train(x = x_train, y = y_train, method = "lm", trControl = cv_5fold) # Addestra LM
rmse_lm <- min(model_lm$results$RMSE) # Estrae metrica

#### 13. Modello 2: Ridge Regression (L2) Tuning ####
# Scopo: Combattere la multicollinearità restringendo proporzionalmente i coefficienti
grid_ridge <- expand.grid(alpha = 0, lambda = seq(0.001, 0.1, length = 20)) # Griglia lambda
model_ridge <- train(x = x_train, y = y_train, method = "glmnet", tuneGrid = grid_ridge, trControl = cv_5fold) # Addestra Ridge
rmse_ridge <- min(model_ridge$results$RMSE) # Estrae metrica

#### 14. Modello 3: Lasso Regression (L1) Tuning ####
# Scopo: Effettuare selezione automatica delle feature azzerando coefficienti ininfluenti
grid_lasso <- expand.grid(alpha = 1, lambda = seq(0.0001, 0.005, length = 20)) # Griglia lambda
model_lasso <- train(x = x_train, y = y_train, method = "glmnet", tuneGrid = grid_lasso, trControl = cv_5fold) # Addestra Lasso
rmse_lasso <- min(model_lasso$results$RMSE) # Estrae metrica

#### 15. Modello 4: Random Forest (Ranger) Tuning ####
# Scopo: Gestire interazioni non lineari e resistere agli outlier mediante ensemble ad alberi
rf_grid <- expand.grid(mtry = c(20, 40, 60), splitrule = "variance", min.node.size = c(3, 5)) # Griglia iperparametri
model_rf <- train(x = x_train, y = y_train, method = "ranger", tuneGrid = rf_grid, trControl = cv_5fold, num.trees = 300) # Addestra Random Forest
rmse_rf <- min(model_rf$results$RMSE) # Estrae metrica

#### 16. Modello 5: XGBoost Tuning ####
# Scopo: Sfruttare gradient boosting per affinare progressivamente i residui
grid_xgb <- expand.grid(nrounds = c(300, 500), max_depth = c(3, 4), eta = c(0.03, 0.05), gamma = 0.01,
                        colsample_bytree = 0.5, min_child_weight = 2, subsample = 0.7) # Griglia iperparametri
model_xgb <- train(x = x_train, y = y_train, method = "xgbTree", tuneGrid = grid_xgb, trControl = cv_5fold, verbosity = 0) # Addestra XGBoost
rmse_xgb <- min(model_xgb$results$RMSE) # Estrae metrica

#### 17. Modello 6: LightGBM Tuning ####
# Scopo: Incrementare l'efficienza predittiva mediante costruzione leaf-wise
dtrain <- lightgbm::lgb.Dataset(data = as.matrix(x_train), label = y_train) # Dataset LGBM
lgb_params <- list(objective = "regression", metric = "rmse", learning_rate = 0.03,
                   num_leaves = 31, feature_fraction = 0.5, bagging_fraction = 0.7, bagging_freq = 1) # Impostazioni
lgb_cv_tune <- lightgbm::lgb.cv(params = lgb_params, data = dtrain, nfold = 5, nrounds = 1000, early_stopping_rounds = 50, verbose = -1) # CV Tuning
rmse_lgb <- lgb_cv_tune$best_score # Estrae metrica
model_lgb <- lightgbm::lgb.train(params = lgb_params, data = dtrain, nrounds = lgb_cv_tune$best_iter, verbose = -1) # Addestra LGBM finale

#### 18. Ensemble Blending e File di Submission ####
# Scopo: Ridurre la varianza combinando i modelli migliori ed esportare per Kaggle
log_lasso <- predict(model_lasso, newdata = x_test) # Inferenza Lasso
log_xgb   <- predict(model_xgb, newdata = x_test)   # Inferenza XGBoost
log_lgb   <- predict(model_lgb, as.matrix(x_test))  # Inferenza LightGBM

log_blend <- (0.15 * log_lasso) + (0.35 * log_xgb) + (0.50 * log_lgb) # Ponderazione logaritmica
dollar_blend <- exp(log_blend) # Reversione in dollari

sub_filename <- generate_sub_filename(SUBMISSION_MODEL_NAME) # Estrae nome file tramite SUBMISSION_MODEL_NAME
submission_df <- tibble(Id = test_df$Id, SalePrice = dollar_blend) # Prepara file
write_csv(submission_df, sub_filename) # Salva file CSV

#### 19. Analisi Grafica e Confronto Modelli ####
# Scopo: Valutare visivamente le prestazioni comparative di tutti gli algoritmi
summary_scores <- tibble(
  Modello = c("LM Baseline", "Ridge", "Lasso", "Random Forest", "XGBoost", "LightGBM", "Triple Blend (Stima)"),
  RMSE_CV = c(rmse_lm, rmse_ridge, rmse_lasso, rmse_rf, rmse_xgb, rmse_lgb, min(rmse_xgb, rmse_lgb) * 0.96)
)

p_eval <- ggplot(summary_scores, aes(x = reorder(Modello, -RMSE_CV), y = RMSE_CV, fill = Modello)) +
  geom_col(show.legend = FALSE, width = 0.55) +
  geom_text(aes(label = round(RMSE_CV, 4)), vjust = -0.5, size = 3.8) +
  scale_fill_manual(values = get_expanded_palette(nrow(summary_scores))) +
  coord_cartesian(ylim = c(0.10, max(summary_scores$RMSE_CV) + 0.02)) +
  labs(
    title = "Comparazione Prestazionale Modelli con Feature di Clustering",
    subtitle = "Tuning Iperparametri ed Ensemble su dati arricchiti da K-Means",
    x = "Architettura di ML",
    y = "RMSE in Cross Validation",
    caption = paste("Fonte dati: Kaggle | Repo:", GITHUB_ADDRESS)
  ) + theme_minimal()

print(p_eval) # Stampa plot finale