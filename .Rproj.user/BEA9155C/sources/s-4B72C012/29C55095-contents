## ----setup, include=FALSE-----------------------------------------------------
knitr::opts_chunk$set(echo = TRUE, cache = TRUE)

# fonte: https://www.kaggle.com/adityadesai13/used-car-dataset-ford-and-mercedes

all_cores <- parallel::detectCores(logical = FALSE)

library(doParallel)
cl <- makePSOCKcluster(all_cores)
registerDoParallel(cl)


## -----------------------------------------------------------------------------
# pacotes necessarios

library(tidyverse)
theme_set(theme_bw())
library(GGally)

# leitura dos dados

vw <- read_csv(file = "dados/vw.csv")


## -----------------------------------------------------------------------------
vw %>%
  select(-model) %>%
  relocate(price) %>%
  relocate(where(is.character), .after = last_col()) %>%
  ggpairs(upper = list(continuous = wrap("cor", method = "spearman")))


## -----------------------------------------------------------------------------
vw %>% 
  group_by(year) %>%
  count() %>%
  ungroup() %>%
  arrange(desc(year)) %>% 
  mutate(percentual = n/sum(n)*100, 
         acumulado = cumsum(percentual)) %>%
  print(n = Inf)


## -----------------------------------------------------------------------------
vw %>% 
  select(year, model) %>%
  group_by(year, model) %>%
  count() %>%
  ungroup() %>%
  complete(year, model, fill = list(n = 0)) %>%
  pivot_wider(names_from = "year", values_from = "n") %>%
  rmarkdown::paged_table()


## -----------------------------------------------------------------------------
vw %>% 
  select(year, model) %>%
  group_by(year, model) %>%
  count() %>%
  ungroup() %>%
  complete(year, model, fill = list(n = 0)) %>%
  mutate(faltante = ifelse(n == 0, "Sim", "Não")) %>%
  ggplot(., aes(x = year, y = model, fill = faltante)) +
  geom_tile(colour = "white") +
  labs(x = "Ano de Fabricação", 
       y = "Modelo do Carro", 
       fill = "Dado faltante?") +
  scale_fill_viridis_d()


## -----------------------------------------------------------------------------
vw_bk <- vw

vw <- 
  vw_bk %>%
  filter(year >= 2015)

vw %>% 
  select(year, model) %>%
  group_by(year, model) %>%
  count() %>%
  ungroup() %>%
  complete(year, model, fill = list(n = 0)) %>%
  mutate(faltante = ifelse(n == 0, "Sim", "Não")) %>%
  ggplot(., aes(x = year, y = model, fill = faltante)) +
  geom_tile(colour = "white") +
  scale_x_continuous(breaks = seq(2015, 2020), labels = seq(2015, 2020)) +
  labs(x = "Ano de Fabricação", 
       y = "Modelo do Carro", 
       fill = "Dado faltante?") +
  scale_fill_viridis_d()


## -----------------------------------------------------------------------------
vw <- 
  vw %>%
  filter(model != "Eos" & model != "Caddy Maxi" & model != "Caddy Life")

vw %>% 
  select(year, model) %>%
  group_by(year, model) %>%
  count() %>%
  ungroup() %>%
  complete(year, model, fill = list(n = 0)) %>%
  mutate(faltante = ifelse(n == 0, "Sim", "Não")) %>%
  ggplot(., aes(x = year, y = model, fill = faltante)) +
  geom_tile(colour = "white") +
  scale_x_continuous(breaks = seq(2015, 2020), labels = seq(2015, 2020)) +
  labs(x = "Ano de Fabricação", 
       y = "Modelo do Carro", 
       fill = "Dado faltante?") +
  scale_fill_viridis_d()


## -----------------------------------------------------------------------------
# pacote necessario

library(tidymodels)

# 75% dos dados como treino

set.seed(2410)

vw_split <- initial_split(vw, prop = .75)

# criar os conjuntos de dados de treino e teste

vw_treino <- training(vw_split)
vw_teste  <- testing(vw_split)


## -----------------------------------------------------------------------------
vw_rec <- 
  # regressao que serah aplicada aos dados
  recipe(price ~ ., 
         data = vw_treino) %>%
  # criacao das variaveis dummy
  step_dummy(where(is.character)) %>%
  # center/scale
  step_center(where(is.numeric), -price) %>% 
  step_scale(where(is.numeric), -price) %>% 
  # funcao para aplicar a transformacao aos dados
  prep()


## -----------------------------------------------------------------------------
# aplicar a transformacao aos dados

vw_treino_t <- juice(vw_rec)

# preparar o conjunto de teste

vw_teste_t <- bake(vw_rec,
                   new_data = vw_teste)


## -----------------------------------------------------------------------------
# modelo

vw_lm <- 
  linear_reg() %>% 
  set_engine("lm")

vw_lm

# criar workflow

vw_wflow <- 
  workflow() %>% 
  add_recipe(vw_rec) %>%
  add_model(vw_lm)

# definicao da validacao cruzada

set.seed(2410)

vw_treino_cv <- vfold_cv(vw_treino, v = 10)

vw_treino_cv

# modelo ajustado com validacao cruzada

vw_lm_fit_cv <- fit_resamples(vw_wflow, vw_treino_cv)

vw_lm_fit_cv

# resultados

collect_metrics(vw_lm_fit_cv)


## -----------------------------------------------------------------------------
# ajuste do modelo no conjunto de treino completo

vw_lm_fit_treino <- fit(vw_wflow, vw_treino)

# resultados no conjunto de teste

resultado <- 
  vw_teste %>%
  bind_cols(predict(vw_lm_fit_treino, vw_teste) %>%
              rename(predicao_lm = .pred))

# resultado final

metrics(resultado, 
        truth = price, 
        estimate = predicao_lm)

# grafico final

ggplot(resultado, aes(x = price, y = predicao_lm)) +
  geom_point() +
  labs(x = "Valores Observados", y = "Valores Preditos") +
  geom_abline(intercept = 0, slope = 1) +
  coord_fixed()


## -----------------------------------------------------------------------------
# definicao do tuning

vw_rf_tune <-
  rand_forest(
    mtry = tune(),
    trees = 1000,
    min_n = tune()
  ) %>%
  set_mode("regression") %>%
  set_engine("ranger", importance = "impurity")

# grid de procura

vw_rf_grid <- grid_regular(mtry(range(1, 9)),
                           min_n(range(10, 50)),
                           levels = c(9, 5))

# workflow

vw_rf_tune_wflow <- 
  workflow() %>%
  add_model(vw_rf_tune) %>%
  add_formula(price ~ .)

# definicao da validacao cruzada

set.seed(2410)

vw_treino_cv <- vfold_cv(vw_treino_t, v = 10)

# avaliacao do modelo

vw_rf_fit_tune <- 
  vw_rf_tune_wflow %>% 
  tune_grid(
    resamples = vw_treino_cv,
    grid = vw_rf_grid
  )

# resultados

collect_metrics(vw_rf_fit_tune)


## -----------------------------------------------------------------------------
vw_rf_fit_tune %>%
  collect_metrics() %>%
  mutate(min_n = factor(min_n)) %>%
  ggplot(., aes(x = mtry, y = mean, colour = min_n, group = min_n)) +
  geom_line() +
  geom_point() +
  facet_grid(~ .metric) +
  scale_x_continuous(breaks = seq(1, 9, 2)) +
  scale_colour_viridis_d()


## -----------------------------------------------------------------------------
# melhores modelos

vw_rf_fit_tune %>%
  show_best("rmse")

# melhor modelo

vw_rf_best <- 
  vw_rf_fit_tune %>%
  select_best("rmse")

vw_rf_final <-
  vw_rf_tune_wflow %>%
  finalize_workflow(vw_rf_best)

vw_rf_final <- fit(vw_rf_final, 
                   vw_treino_t)

vw_rf_final


## -----------------------------------------------------------------------------
# resultados no conjunto de teste

resultado_rf <- 
  vw_teste_t %>%
  bind_cols(predict(vw_rf_final, vw_teste_t) %>%
              rename(predicao_rf = .pred))

metrics(resultado_rf, 
        truth = price, 
        estimate = predicao_rf,
        options = "rmse")

conf_mat(resultado_rf, 
         truth = price, 
         estimate = predicao_rf) %>%
  autoplot(type = "heatmap")

# sensitividade

sens(resultado_rf, 
     truth = price, 
     estimate = predicao_rf)

# especificidade

spec(resultado_rf, 
     truth = price, 
     estimate = predicao_rf)

# importancia das variaveis

vw_rf_final %>% 
  pull_workflow_fit() %>% 
  vip(scale = TRUE)


## -----------------------------------------------------------------------------
# ajuste do modelo no conjunto de treino completo

vw_lm_fit_treino <- fit(vw_wflow, vw_treino)

# resultados no conjunto de teste

resultado <- 
  vw_teste %>%
  bind_cols(predict(vw_lm_fit_treino, vw_teste) %>%
              rename(predicao_lm = .pred))

# resultado final

metrics(resultado, 
        truth = price, 
        estimate = predicao_lm)

# grafico final

ggplot(resultado, aes(x = price, y = predicao_lm)) +
  geom_point() +
  labs(x = "Valores Observados", y = "Valores Preditos") +
  geom_abline(intercept = 0, slope = 1) +
  coord_fixed()

