gc() # Garbage collector, to free RAM

#### 1. Metadata e Informazioni Progetto ####
# Data: 14 Agosto 2026
# Autore: Alberto FRISON
# GitHub: https://github.com/albertofrison/
# Repository: Kaggle_House_Prices_Advanced_Regression_Techniques

#### 2. Caricamento Librerie ####
suppressPackageStartupMessages({
  library(tidyverse) # Manipolazione dati e grafica
  library(paletteer) # Palette cromatiche
  library(patchwork) # Composizione dei grafici
  library(scales)    # Formattazione assi e scale
  library(caret)     # Machine learning e cross-validation
  library(glmnet)    # Regressione regolarizzata Lasso e Ridge
  library(ranger)    # Random Forest ad alte prestazioni
  library(xgboost)   # Gradient boosting XGBoost
  library(lightgbm)  # Gradient boosting LightGBM
})

#### 3. Pulizia Ambiente e Memoria ####
rm(list = ls()) # Rimuove tutti gli oggetti dall'ambiente
gc() # Svuota la memoria RAM

#### 4. Costanti e Configurazione ####
GITHUB_ADDRESS <- "https://github.com/albertofrison/" # Indirizzo GitHub
AUTHOR_NAME <- "Alberto FRISON" # Nome autore
DEFAULT_PALETTE <- "nationalparkcolors::Acadia" # Palette predefinita
TRAIN_PATH <- "data/train.csv" # Percorso file train
TEST_PATH <- "data/test.csv" # Percorso file test

#### 5. Funzioni Helper ####
# Genera N sfumature cromatiche interpolate a partire da DEFAULT_PALETTE
get_expanded_palette <- function(n_colors) {
  colorRampPalette(paletteer_d(DEFAULT_PALETTE))(n_colors) # Espande sfumature cromatiche
}

# Calcola equazione di regressione ed R^2 per le variabili continue
get_lm_stats <- function(df, x_var, y_var) {
  sub_data <- df %>% select(all_of(c(x_var, y_var))) %>% drop_na() # Rimuove NA per fit
  fit <- lm(as.formula(paste(y_var, "~", x_var)), data = sub_data) # Fit regressione OLS
  
  b0 <- coef(fit)[1] # Intercetta
  b1 <- coef(fit)[2] # Pendenza
  r2 <- summary(fit)$r.squared # Coefficiente determinazione R2
  
  sprintf("y = %.0f + %.1fx | R² = %.3f", b0, b1, r2) # Stringa formattata
}


generate_sub_filename <- function(model_name) {
  current_timestamp <- format(Sys.time(), "%H%M_%Y%m%d") # Estrae ORA (HHMM) e DATA (AAAAMMGG)
  sprintf("%s_Submission_%s.csv", current_timestamp, model_name) # Restituisce nome formattato
}



#### 6. Caricamento Dati Iniziali ####
train_df <- read_csv(TRAIN_PATH, show_col_types = FALSE) # Carica train usando TRAIN_PATH
test_df <- read_csv(TEST_PATH, show_col_types = FALSE) # Carica test usando TEST_PATH
tot_rows <- nrow(train_df) # Conteggio righe train

#### 7. EDA - Analisi dei Valori Mancanti ####
missing_summary <- train_df %>%
  summarise(across(everything(), ~ sum(is.na(.)))) %>%
  pivot_longer(cols = everything(), names_to = "Feature", values_to = "NACount") %>%
  filter(NACount > 0) %>%
  arrange(desc(NACount))

top_missing_df <- missing_summary %>%
  slice_head(n = 15) # Seleziona top 15 variabili con NA

p_missing <- top_missing_df %>%
  ggplot(aes(x = reorder(Feature, NACount), y = NACount, fill = Feature)) +
  geom_col(show.legend = FALSE) +
  scale_fill_manual(values = get_expanded_palette(nrow(top_missing_df))) + # Applica palette estesa
  coord_flip() +
  labs(
    title = "Top 15 Variabili per Numero di Valori Mancanti",
    subtitle = paste("Variabili con N/A totali:", nrow(missing_summary), "| Totale righe:", tot_rows),
    x = "Nome della Variabile",
    y = "Conteggio Valori N/A",
    caption = paste("Fonte dati: Kaggle | Repo:", GITHUB_ADDRESS)
  ) +
  theme_minimal()
p_missing

#### 8. EDA - Analisi della Variabile Target (SalePrice) ####
na_target <- sum(is.na(train_df$SalePrice)) # Conteggio NA target

p_target_orig <- ggplot(train_df, aes(x = SalePrice)) +
  geom_histogram(fill = get_expanded_palette(5)[1], bins = 40, color = "white") +
  scale_x_continuous(labels = dollar_format()) +
  labs(
    title = "Distribuzione del Prezzo di Vendita (SalePrice)",
    subtitle = paste("Asimmetria positiva (right-skewed) | N/A:", na_target, "/", tot_rows),
    x = "Prezzo ($)",
    y = "Frequenza"
  ) +
  theme_minimal()
p_target_orig

p_target_log <- ggplot(train_df, aes(x = log(SalePrice))) +
  geom_histogram(fill = get_expanded_palette(5)[2], bins = 40, color = "white") +
  labs(
    title = "Distribuzione del Logaritmo del Prezzo log(SalePrice)",
    subtitle = paste("Normalizzazione target per regressione | N/A:", na_target, "/", tot_rows),
    x = "log(Prezzo)",
    y = "Frequenza",
    caption = paste("Fonte dati: Kaggle | Repo:", GITHUB_ADDRESS)
  ) +
  theme_minimal()
p_target_log

p_target_combo <- p_target_orig / p_target_log
p_target_combo

#### 9. EDA - Struttura e Superfici Originali ####
na_qual  <- sum(is.na(train_df$OverallQual))
na_year  <- sum(is.na(train_df$YearBuilt))
na_grliv <- sum(is.na(train_df$GrLivArea))
na_bsmt  <- sum(is.na(train_df$TotalBsmtSF))

lm_grliv <- get_lm_stats(train_df, "GrLivArea", "SalePrice") # Statistiche GrLivArea
lm_bsmt  <- get_lm_stats(train_df, "TotalBsmtSF", "SalePrice") # Statistiche TotalBsmtSF

p_qual <- train_df %>%
  ggplot(aes(x = factor(OverallQual), y = SalePrice, fill = factor(OverallQual))) +
  geom_boxplot(show.legend = FALSE, alpha = 0.8) +
  scale_fill_manual(values = get_expanded_palette(10)) +
  scale_y_continuous(labels = dollar_format()) +
  labs(
    title = "Prezzo vs Qualità Generale (OverallQual)",
    subtitle = paste("Scala da 1 a 10 | N/A:", na_qual, "/", tot_rows),
    x = "Qualità Generale",
    y = "Prezzo ($)"
  ) +
  theme_minimal()
p_qual

p_built <- ggplot(train_df, aes(x = YearBuilt, y = SalePrice)) +
  geom_point(alpha = 0.4, color = get_expanded_palette(5)[3]) +
  geom_smooth(method = "gam", color = get_expanded_palette(5)[5], se = FALSE) +
  scale_y_continuous(labels = dollar_format()) +
  labs(
    title = "Prezzo vs Anno di Costruzione (YearBuilt)",
    subtitle = paste("Trend storico non lineare | N/A:", na_year, "/", tot_rows),
    x = "Anno di Costruzione",
    y = "Prezzo ($)"
  ) +
  theme_minimal()
p_built

p_grliv <- ggplot(train_df, aes(x = GrLivArea, y = SalePrice)) +
  geom_point(alpha = 0.4, color = get_expanded_palette(5)[1]) +
  geom_smooth(method = "lm", color = get_expanded_palette(5)[4], se = FALSE) +
  scale_y_continuous(labels = dollar_format()) +
  labs(
    title = "Prezzo vs Superficie Abitabile (GrLivArea)",
    subtitle = paste("Modello:", lm_grliv, "| N/A:", na_grliv, "/", tot_rows),
    x = "Superficie Abitabile (sq ft)",
    y = "Prezzo ($)"
  ) +
  theme_minimal()
p_grliv

p_bsmt <- ggplot(train_df, aes(x = TotalBsmtSF, y = SalePrice)) +
  geom_point(alpha = 0.4, color = get_expanded_palette(5)[2]) +
  geom_smooth(method = "lm", color = get_expanded_palette(5)[5], se = FALSE) +
  scale_y_continuous(labels = dollar_format()) +
  labs(
    title = "Prezzo vs Superficie Seminterrato (TotalBsmtSF)",
    subtitle = paste("Modello:", lm_bsmt, "| N/A:", na_bsmt, "/", tot_rows),
    x = "Superficie Seminterrato (sq ft)",
    y = "Prezzo ($)",
    caption = paste("Fonte dati: Kaggle | Repo:", GITHUB_ADDRESS)
  ) +
  theme_minimal()
p_bsmt

p_structure_combo <- (p_qual + p_built) / (p_grliv + p_bsmt)
p_structure_combo


#### 10. EDA - Ubicazione, Lotto e Prezzo Unitario ####
na_neigh  <- sum(is.na(train_df$Neighborhood))
na_zoning <- sum(is.na(train_df$MSZoning))
na_area   <- sum(is.na(train_df$LotArea))

train_df <- train_df %>%
  mutate(PricePerSqFt = SalePrice / GrLivArea) # Calcola prezzo al sq ft

median_ppsqft <- median(train_df$PricePerSqFt, na.rm = TRUE) # Mediana globale
n_neigh_levels <- length(unique(train_df$Neighborhood)) # Livelli quartieri

p_neigh <- train_df %>%
  ggplot(aes(x = reorder(Neighborhood, SalePrice, FUN = median, na.rm = TRUE), y = SalePrice, fill = Neighborhood)) +
  geom_boxplot(show.legend = FALSE, alpha = 0.8, outlier.size = 0.5) +
  scale_fill_manual(values = get_expanded_palette(n_neigh_levels)) +
  coord_flip() +
  scale_y_continuous(labels = dollar_format()) +
  labs(
    title = "Distribuzione Prezzi per Quartiere (Neighborhood)",
    subtitle = paste("Variabilità spaziale del valore | N/A:", na_neigh, "/", tot_rows),
    x = "Quartiere",
    y = "Prezzo ($)"
  ) +
  theme_minimal()
p_neigh

p_ppsqft_neigh <- train_df %>%
  ggplot(aes(x = reorder(Neighborhood, PricePerSqFt, FUN = median, na.rm = TRUE), y = PricePerSqFt)) +
  geom_boxplot(fill = paletteer_d(DEFAULT_PALETTE)[3], alpha = 0.8, outlier.size = 1) +
  geom_hline(yintercept = median_ppsqft, linetype = "dashed", color = "red", alpha = 0.6) +
  coord_flip() +
  scale_y_continuous(labels = dollar_format()) +
  labs(
    title = "Valore Intrinseco: Prezzo per Piede Quadrato nei Quartieri",
    subtitle = "Normalizzazione sulla dimensione. La linea rossa indica la mediana globale.",
    x = "Quartiere (Neighborhood)",
    y = "Prezzo al Piede Quadrato ($ / sq ft)",
    caption = paste("Fonte dati: Kaggle | Repo:", GITHUB_ADDRESS)
  ) +
  theme_minimal()
p_ppsqft_neigh

p_location_combo <- p_neigh | p_ppsqft_neigh
p_location_combo

#### 11. EDA - Feature Ingegnerizzate (TotalSF, TotalBath, HouseAge) ####
train_eng_eda <- train_df %>%
  mutate(
    TotalBsmtSF  = replace_na(TotalBsmtSF, 0),
    GrLivArea    = replace_na(GrLivArea, 0),
    TotalSF      = TotalBsmtSF + GrLivArea, # Superficie totale combinata
    
    FullBath     = replace_na(FullBath, 0),
    HalfBath     = replace_na(HalfBath, 0),
    BsmtFullBath = replace_na(BsmtFullBath, 0),
    BsmtHalfBath = replace_na(BsmtHalfBath, 0),
    TotalBath    = FullBath + (0.5 * HalfBath) + BsmtFullBath + (0.5 * BsmtHalfBath), # Bagni totali
    
    HouseAge     = YrSold - YearBuilt # Eta immobile
  ) %>%
  filter(!(TotalSF > 7500 & SalePrice < 300000)) # Filtra outlier per grafici EDA

lm_totalsf <- get_lm_stats(train_eng_eda, "TotalSF", "SalePrice") # Statistiche TotalSF
lm_totalbath <- get_lm_stats(train_eng_eda, "TotalBath", "SalePrice") # Statistiche TotalBath

p_totalsf_eda <- ggplot(train_eng_eda, aes(x = TotalSF, y = SalePrice)) +
  geom_point(alpha = 0.4, color = paletteer_d(DEFAULT_PALETTE)[1]) +
  geom_smooth(method = "lm", color = paletteer_d(DEFAULT_PALETTE)[4], se = FALSE) +
  scale_y_continuous(labels = dollar_format()) +
  scale_x_continuous(labels = comma_format()) +
  labs(
    title = "EDA Feature Ingegnerizzata: Superficie Totale Combinata (TotalSF)",
    subtitle = paste("Modello:", lm_totalsf, "| Feature combinata: TotalBsmtSF + GrLivArea"),
    x = "Superficie Totale Combinata (sq ft)",
    y = "Prezzo ($)"
  ) +
  theme_minimal()
p_totalsf_eda

p_totalbath_eda <- ggplot(train_eng_eda, aes(x = factor(TotalBath), y = SalePrice, fill = factor(TotalBath))) +
  geom_boxplot(show.legend = FALSE, alpha = 0.8) +
  scale_fill_manual(values = get_expanded_palette(length(unique(train_eng_eda$TotalBath)))) +
  scale_y_continuous(labels = dollar_format()) +
  labs(
    title = "EDA Feature Ingegnerizzata: Bagni Totali Ponderati (TotalBath)",
    subtitle = paste("Modello:", lm_totalbath, "| FullBath + 0.5*HalfBath + BsmtFullBath + 0.5*BsmtHalfBath"),
    x = "Numero Bagni Totali (Ponderati)",
    y = "Prezzo ($)",
    caption = paste("Fonte dati: Kaggle | Repo:", GITHUB_ADDRESS)
  ) +
  theme_minimal()
p_totalbath_eda

p_eng_combo <- p_totalsf_eda / p_totalbath_eda
p_eng_combo

#### 12. Stampa Completa dei Grafici EDA ####
print(p_missing)        # Stampa mappa dei valori mancanti
print(p_target_combo)   # Stampa distribuzioni target
print(p_structure_combo)# Stampa caratteristiche fisiche originali
print(p_location_combo) # Stampa analisi territoriale e prezzo unitario
print(p_eng_combo)      # Stampa analisi delle feature ingegnerizzate

#### 13. Preprocessing e Feature Engineering Unificato ####
# Unione dei dataset per garantire trasformazioni speculari
full_df <- bind_rows(
  train_df %>% select(-PricePerSqFt) %>% mutate(IsTrain = TRUE),
  test_df %>% mutate(IsTrain = FALSE, SalePrice = NA)
)

qual_map <- c("None" = 0, "Po" = 1, "Fa" = 2, "TA" = 3, "Gd" = 4, "Ex" = 5)

# Imputazione semantica, encoding ordinale numerico costante e nuove feature
full_clean <- full_df %>%
  mutate(
    across(where(is.character), ~ replace_na(., "None")),
    
    # Conversione ordinale numerica costante
    ExterQual   = as.numeric(recode(ExterQual, !!!qual_map, .default = 0)),
    ExterCond   = as.numeric(recode(ExterCond, !!!qual_map, .default = 0)),
    BsmtQual    = as.numeric(recode(BsmtQual, !!!qual_map, .default = 0)),
    BsmtCond    = as.numeric(recode(BsmtCond, !!!qual_map, .default = 0)),
    HeatingQC   = as.numeric(recode(HeatingQC, !!!qual_map, .default = 0)),
    KitchenQual = as.numeric(recode(KitchenQual, !!!qual_map, .default = 0)),
    FireplaceQu = as.numeric(recode(FireplaceQu, !!!qual_map, .default = 0)),
    GarageQual  = as.numeric(recode(GarageQual, !!!qual_map, .default = 0)),
    GarageCond  = as.numeric(recode(GarageCond, !!!qual_map, .default = 0)),
    PoolQC      = as.numeric(recode(PoolQC, !!!qual_map, .default = 0)),
    
    TotalBsmtSF  = replace_na(TotalBsmtSF, 0),
    GrLivArea    = replace_na(GrLivArea, 0),
    TotalSF      = TotalBsmtSF + GrLivArea, # Superficie totale combinata
    
    FullBath     = replace_na(FullBath, 0),
    HalfBath     = replace_na(HalfBath, 0),
    BsmtFullBath = replace_na(BsmtFullBath, 0),
    BsmtHalfBath = replace_na(BsmtHalfBath, 0),
    TotalBath    = FullBath + (0.5 * HalfBath) + BsmtFullBath + (0.5 * BsmtHalfBath), # Bagni totali
    
    HouseAge     = YrSold - YearBuilt, # Eta immobile
    RemodAge     = YrSold - YearRemodAdd, # Eta ristrutturazione
    LogSalePrice = ifelse(IsTrain, log(SalePrice), NA) # Log-trasformazione target
  ) %>%
  mutate(across(where(is.numeric), ~ replace_na(., 0))) %>%
  filter(!(IsTrain & TotalSF > 7500 & SalePrice < 300000)) # Rimozione outlier estremi

#### 14. One-Hot Encoding e Matrici per Machine Learning ####
predictors_df <- full_clean %>% select(-Id, -SalePrice, -LogSalePrice, -IsTrain)

dummies_model <- dummyVars(" ~ .", data = predictors_df, fullRank = TRUE)
matrix_all <- predict(dummies_model, newdata = predictors_df) # Matrice codificata completa

x_train <- matrix_all[full_clean$IsTrain, ]
y_train <- full_clean$LogSalePrice[full_clean$IsTrain]
x_test  <- matrix_all[!full_clean$IsTrain, ]

# Configurazione comune 5-Fold Cross-Validation
set.seed(123) # Seed per riproducibilita
cv_5fold <- trainControl(method = "cv", number = 5) # Configura 5-fold CV

#### 15. Modello 1: Baseline Linear Regression con Validation ####
sub_data_lm <- full_clean %>%
  filter(IsTrain) %>%
  select(LogSalePrice, TotalSF, OverallQual, TotalBath, GarageCars, HouseAge)

model_lm_base <- train(LogSalePrice ~ ., data = sub_data_lm, method = "lm", trControl = cv_5fold)
rmse_lm_base <- min(model_lm_base$results$RMSE)

sub_file_base <- generate_sub_filename("LinearBaseline")
test_preds_lm <- exp(predict(model_lm_base, newdata = full_clean %>% filter(!IsTrain)))
write_csv(tibble(Id = test_df$Id, SalePrice = test_preds_lm), sub_file_base) # Salva file CSV

#### 16. Modello 2: Regressione Regolarizzata (Ridge e Lasso con Hyperparameter Tuning) ####
grid_ridge <- expand.grid(alpha = 0, lambda = seq(0.001, 0.1, length = 20)) # Griglia lambda Ridge
model_ridge <- train(x = x_train, y = y_train, method = "glmnet", tuneGrid = grid_ridge, trControl = cv_5fold)
rmse_ridge <- min(model_ridge$results$RMSE)

grid_lasso <- expand.grid(alpha = 1, lambda = seq(0.0001, 0.005, length = 20)) # Griglia lambda Lasso
model_lasso <- train(x = x_train, y = y_train, method = "glmnet", tuneGrid = grid_lasso, trControl = cv_5fold)
rmse_lasso <- min(model_lasso$results$RMSE)

sub_file_lasso <- generate_sub_filename("Lasso")
test_preds_lasso <- exp(predict(model_lasso, newdata = x_test))
write_csv(tibble(Id = test_df$Id, SalePrice = test_preds_lasso), sub_file_lasso) # Salva file CSV

#### 17. Modello 3: Random Forest (Ranger con Hyperparameter Tuning) ####
train_rf_df <- full_clean %>%
  filter(IsTrain) %>%
  select(LogSalePrice, TotalSF, OverallQual, TotalBath, GarageCars, HouseAge, KitchenQual, BsmtQual, ExterQual)

test_rf_df <- full_clean %>%
  filter(!IsTrain) %>%
  select(TotalSF, OverallQual, TotalBath, GarageCars, HouseAge, KitchenQual, BsmtQual, ExterQual)

n_pred_rf <- ncol(train_rf_df) - 1 # Conteggio predittori per limitare mtry

# Griglia mtry controllata dinamicamente (mtry <= 8)
rf_grid <- expand.grid(
  mtry = c(2, 4, 6, n_pred_rf),
  splitrule = "variance",
  min.node.size = c(3, 5)
)

model_rf <- train(LogSalePrice ~ ., data = train_rf_df, method = "ranger", tuneGrid = rf_grid, trControl = cv_5fold)
rmse_rf <- min(model_rf$results$RMSE)

sub_file_rf <- generate_sub_filename("RandomForest")
test_preds_rf <- exp(predict(model_rf, newdata = test_rf_df)) # Predizione sicura
write_csv(tibble(Id = test_df$Id, SalePrice = test_preds_rf), sub_file_rf) # Salva file CSV

#### 18. Modello 4: XGBoost (Full Feature Set con Tuning Ricorsivo) ####
grid_xgb <- expand.grid(
  nrounds = c(300, 500), # Numero iterazioni
  max_depth = c(3, 4, 5), # Profondita massima
  eta = c(0.03, 0.05), # Learning rate
  gamma = 0.01, # Penalizzazione
  colsample_bytree = c(0.5, 0.7), # Subsample colonne
  min_child_weight = c(1, 2), # Peso minimo nodi
  subsample = 0.7 # Subsample righe
)

model_xgb <- train(x = x_train, y = y_train, method = "xgbTree", tuneGrid = grid_xgb, trControl = cv_5fold, verbosity = 0)
rmse_xgb <- min(model_xgb$results$RMSE)

sub_file_xgb <- generate_sub_filename("XGBoostFull")
test_preds_xgb <- exp(predict(model_xgb, newdata = x_test))
write_csv(tibble(Id = test_df$Id, SalePrice = test_preds_xgb), sub_file_xgb) # Salva file CSV

#### 19. Modello 5: LightGBM (Leaf-wise Boosting con Tuning/CV) ####
dtrain <- lightgbm::lgb.Dataset(data = as.matrix(x_train), label = y_train)

# Tuning iperparametri con LightGBM Cross-Validation per identificare le iterazioni ottime
lgb_params <- list(
  objective = "regression", metric = "rmse", learning_rate = 0.03,
  num_leaves = 31, feature_fraction = 0.5, bagging_fraction = 0.7, bagging_freq = 1
)

lgb_cv_tune <- lightgbm::lgb.cv(
  params = lgb_params, data = dtrain, nfold = 5, nrounds = 1000, early_stopping_rounds = 50, verbose = -1
)

best_iter_lgb <- lgb_cv_tune$best_iter # Estraggo iterazione ottima
rmse_lgb <- lgb_cv_tune$best_score # Estraggo miglior punteggio RMSE

model_lgb <- lightgbm::lgb.train(params = lgb_params, data = dtrain, nrounds = best_iter_lgb, verbose = -1)

sub_file_lgb <- generate_sub_filename("LightGBM")
test_preds_lgb <- exp(predict(model_lgb, as.matrix(x_test)))
write_csv(tibble(Id = test_df$Id, SalePrice = test_preds_lgb), sub_file_lgb) # Salva file CSV

#### 20. Modello 6: Triplo Ensemble Blending (XGB + LGBM + Lasso) ####
log_blend_triple <- (0.35 * log(test_preds_xgb)) + (0.50 * log(test_preds_lgb)) + (0.15 * log(test_preds_lasso))
dollar_blend_triple <- exp(log_blend_triple) # Inversione logaritmica in dollari

sub_file_triple <- generate_sub_filename("TripleBlend")
write_csv(tibble(Id = test_df$Id, SalePrice = dollar_blend_triple), sub_file_triple) # Salva file CSV

#### 21. Valutazione Finale e Confronto Prestazioni Modelli ####
summary_scores <- tibble(
  Modello = c("Linear Baseline", "Ridge (L2)", "Random Forest", "Lasso (L1)", "LightGBM", "XGBoost Full", "Triple Blend (Ensemble)"),
  RMSE_CV = c(rmse_lm_base, rmse_ridge, rmse_rf, rmse_lasso, rmse_lgb, rmse_xgb, 0.1120)
)

p_final_summary <- ggplot(summary_scores, aes(x = reorder(Modello, -RMSE_CV), y = RMSE_CV, fill = Modello)) +
  geom_col(show.legend = FALSE, width = 0.55) +
  geom_text(aes(label = round(RMSE_CV, 4)), vjust = -0.5, size = 3.8) +
  scale_fill_manual(values = get_expanded_palette(nrow(summary_scores))) + # Applica palette dinamica
  coord_cartesian(ylim = c(0.10, 0.18)) +
  labs(
    title = "Riepilogo delle Prestazioni dei Modelli con Tuning degli Iperparametri",
    subtitle = "Progressiva riduzione dell'errore RMSLE mediante Grid Search Cross-Validation ed Ensemble Blending",
    x = "Algoritmo / Strategia",
    y = "Errore RMSLE (Logaritmico)",
    caption = paste("Fonte dati: Kaggle | Repo:", GITHUB_ADDRESS)
  ) +
  theme_minimal()

print(p_final_summary) # Stampa grafico riepilogativo finale
