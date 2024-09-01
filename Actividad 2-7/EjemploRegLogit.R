
rm(list=ls())
# Configurar el entorno
set.seed(123)
library(ggplot2)

# Generar datos simulados
n <- 500  # Número de observaciones
Edad <- rnorm(n, mean = 35, sd = 10)  # Edad del cliente
Ingreso <- rnorm(n, mean = 50000, sd = 15000)  # Ingreso del cliente

# Generar la variable dependiente (Compra) usando una función logística
p <- 1 / (1 + exp(-(-3 + 0.05 * Edad + 0.00005 * Ingreso)))  # Probabilidad de compra
Compra <- rbinom(n, size = 1, prob = p)  # Variable dependiente binaria

# Crear el data frame
datos <- data.frame(Edad, Ingreso, Compra)

# Ajustar el modelo de regresión logística
modelo <- glm(Compra ~ Edad + Ingreso, data = datos, family = binomial)

# Resumen del modelo
summary(modelo)

# Ajustar el modelo nulo
modelo_nulo <- glm(Compra ~ 1, data = datos, family = binomial)

# Comparar los modelos con anova
anova_resultado <- anova(modelo_nulo, modelo, test = "Chi")
print(anova_resultado)

# Coeficientes del modelo
coeficientes <- summary(modelo)$coefficients
print(coeficientes)

# Calcular las probabilidades predichas
datos$predicciones <- predict(modelo, type = "response")

# Graficar las predicciones vs Edad
ggplot(datos, aes(x = Edad, y = predicciones)) +
  geom_point(aes(color = as.factor(Compra)), alpha = 0.5) +
  labs(title = "Probabilidad Predicha de Compra vs Edad",
       x = "Edad",
       y = "Probabilidad de Compra") +
  theme_minimal()

# Convertir las predicciones en una clase (0 o 1) con un umbral de 0.5
datos$predicciones_clase <- ifelse(datos$predicciones > 0.5, 1, 0)

# Crear una tabla de contingencia
tabla_contingencia <- table(Predicciones = datos$predicciones_clase, Real = datos$Compra)
print(tabla_contingencia)

# Calcular las métricas
tasa_error <- 1 - sum(diag(tabla_contingencia)) / sum(tabla_contingencia)
sensibilidad <- tabla_contingencia[2, 2] / sum(tabla_contingencia[2, ])
especificidad <- tabla_contingencia[1, 1] / sum(tabla_contingencia[1, ])

cat("Tasa de Error:", tasa_error, "\n")
cat("Sensibilidad:", sensibilidad, "\n")
cat("Especificidad:", especificidad, "\n")

# Gráfico de residuos
plot(modelo$residuals ~ datos$Edad, main = "Residuos vs Edad",
     xlab = "Edad", ylab = "Residuos")
abline(h = 0, col = "red")

# Gráfico Logit vs Edad
logit_predicciones <- log(datos$predicciones / (1 - datos$predicciones))
plot(logit_predicciones ~ datos$Edad, main = "Logit vs Edad",
     xlab = "Edad", ylab = "Logit")
abline(lm(logit_predicciones ~ datos$Edad), col = "blue")



###############################################
################################################
##### Ejemplo con más de dos niveles en var respuesta
###########################################
rm(list=ls())
# Cargar librerías necesarias
# Cargar librerías necesarias
library(nnet)
library(ggplot2)

# Configurar la semilla para reproducibilidad
set.seed(123)

# Generar datos simulados
n <- 600  # Número de observaciones
X1 <- rnorm(n, mean = 50, sd = 10)  # Predictor X1
X2 <- rnorm(n, mean = 30, sd = 5)   # Predictor X2

# Generar valores logit para las categorías
logit_A <- 0.05 * X1 + 0.02 * X2 - 1
logit_B <- 0.03 * X1 - 0.01 * X2 - 0.5

# Convertir logits a probabilidades usando softmax
exp_logit_A <- exp(logit_A)
exp_logit_B <- exp(logit_B)
exp_logit_C <- 1  # Logit para la categoría C se considera 0

# Normalizar las probabilidades
denominator <- exp_logit_A + exp_logit_B + exp_logit_C
p1 <- exp_logit_A / denominator
p2 <- exp_logit_B / denominator
p3 <- exp_logit_C / denominator

# Asignar categorías basadas en las probabilidades
categoria <- sapply(1:n, function(i) {
  sample(c("A", "B", "C"), size = 1, prob = c(p1[i], p2[i], p3[i]))
})

# Crear el data frame
datos <- data.frame(X1, X2, Categoria = as.factor(categoria))


# Ajustar el modelo de regresión logística multinomial
modelo_multinomial <- multinom(Categoria ~ X1 + X2, data = datos)

# Resumen del modelo
summary(modelo_multinomial)

# Coeficientes del modelo
coeficientes <- summary(modelo_multinomial)$coefficients
print(coeficientes)

# Calcular las probabilidades predichas
datos$predicciones <- predict(modelo_multinomial, type = "probs")

# Mostrar las primeras filas de las predicciones
head(datos$predicciones)

# Ajustar el modelo nulo
modelo_nulo <- multinom(Categoria ~ 1, data = datos)

# Comparar los modelos con anova
anova_resultado <- anova(modelo_nulo, modelo_multinomial)
print(anova_resultado)

# Convertir las predicciones en un data frame para graficar
datos_pred <- as.data.frame(datos$predicciones)
colnames(datos_pred) <- c("Prob_A", "Prob_B", "Prob_C")
datos_pred$Categoria <- datos$Categoria

# Graficar las probabilidades predichas para cada categoría
ggplot(datos_pred, aes(x = X1, y = Prob_A, color = Categoria)) +
  geom_point(alpha = 0.5) +
  labs(title = "Probabilidad Predicha para la Categoría A vs X1",
       x = "X1",
       y = "Probabilidad de Categoría A") +
  theme_minimal()

ggplot(datos_pred, aes(x = X2, y = Prob_B, color = Categoria)) +
  geom_point(alpha = 0.5) +
  labs(title = "Probabilidad Predicha para la Categoría B vs X2",
       x = "X2",
       y = "Probabilidad de Categoría B") +
  theme_minimal()


###############################################
##############################################
##### Ejemplo con var indep categórica


rm(list=ls())

# Simular datos
set.seed(123)
n <- 100  # Número de observaciones

# Variable dependiente binaria
y <- rbinom(n, 1, 0.5)

# Variable independiente continua
x1 <- rnorm(n)

# Variable independiente categórica (con 3 niveles)
x2 <- factor(sample(c("A", "B", "C"), n, replace = TRUE))

# Crear un data frame con los datos simulados
data <- data.frame(y, x1, x2)

# Convertir la variable categórica en variables dummy
data$x2 <- relevel(data$x2, ref = "A")  # Establecer "A" como referencia

# Ajustar el modelo de regresión logística
modelo <- glm(y ~ x1 + x2, data = data, family = binomial)

# Resumen del modelo
summary(modelo)

# Validación del modelo
# Predicciones de probabilidad
predicciones <- predict(modelo, type = "response")

# Curva ROC y AUC
library(pROC)
roc_obj <- roc(data$y, predicciones)
plot(roc_obj, main = "Curva ROC")
auc(roc_obj)

# Matriz de confusión
pred_clas <- ifelse(predicciones > 0.5, 1, 0)
table(Predicted = pred_clas, Actual = data$y)

# Gráfica de predicciones vs valores reales
plot(data$y, predicciones, main = "Predicciones vs Valores Reales",
     xlab = "Valores Reales", ylab = "Predicciones")
abline(0, 1, col = "red")

