# Exemplo de Importance Sampling
#
# Distribuição alvo: Normal absoluta
#   f(x) = exp(-x^2/2)/Zf
#      Zf = sqrt(pi/2)
#
# Função f_tilde (não-normalizada):
#   f_tilde(x) = exp(-x^2/2)
#
# Distribuição proposta (normalizada): Exp(lam)
#   g(x) = 2*exp(-2*x)


set.seed(42)
N <- 100000 # tamanho da amostra

# Disribuição alvo (normalizada)
f <- function(x){
    y <- sqrt(2/pi)*exp(-x^2/2)
    return(y)
}

# Distribuição alvo não-normalizada
f_tilde <- function(x){
    y <- exp(-x^2/2)
    return(y)
}

# Função auxiliar f(x)*x para calcular a esperança por integração
fx <- function(x){
    y <- sqrt(2/pi)*exp(-x^2/2)*x
    return(y)
}

# Densidade da distribuição proposta
g <- function(x){
	y <- dexp(x)
	return(y)
	}

# Função dos pesos não-normalizados
w_tilde <- function(x){
    y <- 1/2*exp(-x^2/2 + 2*x)
    return(y)
}

#####
# Compara as densidades alvo e propostas
x_ = seq(0,4,0.1)
plot(x_,f(x_), type="l", xlab="x", ylab="densidade")
lines(x_, g(x_), col="red")
legend(x="topright", legend=c("f(x)", "g(x)"), lty=1, col=c("black", "red"))

#####
# Algortimo IS:
# 1. Gera uma amostra de g(x) de tamanho N
X <- rexp(N, rate=2)

# 2. calcula os pesos não-normalizados
W_tilde <- w_tilde(X)

# 3. Normaliza os pesos
W <- W_tilde/sum(W_tilde)

# a) Cálculo do valor esperado de X por IS:
muX <- sum(X*W)
cat("\nE[X] calculado por IS:", muX)

# Cálculo por integração numérica:
muX_int <- integrate(fx, 0, Inf)$value
cat("\nE[X] calculado por integração numérica:", muX_int)

# b) Cálculo da constante normalizadora por IS
Zf <- mean(W_tilde)
cat("\nZf calculado por IS:", Zf)

# Compara Zf com o valor verdadeiro:
Zf_true <- sqrt(pi/2)
cat("\nZf verdadeiro:", Zf_true)

#####
# Reamostragem
Xf <- sample(X, N, replace=TRUE, prob=W)
x_ <- seq(0,4,0.1)

# Compara as amostras obtidas por reamostragem (histograma) com a densidade alvo
hist(Xf, freq=FALSE, xlab="x", ylab="densidade", main="", breaks=20, xlim=range(0,4))
lines(x_, f(x_), col='red')
legend(x="topright", legend=c("f(x)"), lty=1, col="red")