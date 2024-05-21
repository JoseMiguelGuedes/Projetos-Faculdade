###############################################################################################################################
###########                                           LIBRARY                                                       ###########
###############################################################################################################################
df <- read.csv("/Users/joseguedes/Documents/Mestrado em gestão/2ºsemestre/Regression Analysis and Multivariate Data Analysis/trabalho/GrowthDJ.csv")

df <- df[,2:11] ## Removal of columns containing irrelevant information

library(ggplot2)
library(car)
library(corrplot)
library(fastDummies)
library(lmtest)
library(sandwich)
library(nlme)
library(ggpubr)
library(moments)
library(DescTools)
library(olsrr)

categoricas <- c(1,2,3)
numericas <- c(4,5,6,7,8,9,10)

for (i in numericas){
  df[,i] <- as.numeric(df[,i])
}

for (i in categoricas){
  df[,i] <- as.factor(df[,i])
}

str(df)

###############################################################################################################################
###########                                              NA´S                                                       ###########
###############################################################################################################################

str(df)
 
df <- df[complete.cases(df), ] ## GDP Growth will be our dependent variable. Removal of all NA values

rownames(df) <- NULL ## Resetting the index

###############################################################################################################################
###########                                          Data Analysis                                                  ###########
###############################################################################################################################

str(df)
summary(df)

apply(df[,numericas],2,sd)

IQR <- function(x) 
{
  quantile(x,0.75) - quantile(x,0.25)  
}
apply(df[,numericas],2,IQR)

###############################################################################################################################
###########                                           Correlation                                                   ###########
###############################################################################################################################

cor(df[,numericas])
corrplot(corr = cor(df[,numericas]), method = 'number', type = "lower")

###############################################################################################################################
###########                                            Dummies                                                      ###########
###############################################################################################################################

### Graphs - Boxplots ###

ggplot(df,aes(x=oil,y=gdpgrowth)) + 
  geom_boxplot(aes(fill = oil)) +
  labs(x="oil Production", y="GDP Growth") +
  theme_bw() + 
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank()) 

ggplot(df,aes(x=inter,y=gdpgrowth)) + 
  geom_boxplot(aes(fill = inter)) +
  labs(x="Better Data Quality", y="GDP Growth") +
  theme_bw() + 
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank()) 

ggplot(df,aes(x=oecd,y=gdpgrowth)) + 
  geom_boxplot(aes(fill = oecd)) +
  labs(x="OECD", y="GDP Growth") +
  theme_bw() + 
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank()) 

##### Graphs - Others #####

graficooil <- ggplot(df,aes(x=as.numeric(oil),y=gdpgrowth)) +  geom_point() + geom_smooth(method=lm) +
  ggtitle("GDP Growth vs. oil Production") + theme(plot.title=element_text(hjust=0.5)) + xlab("oil")

graficointer <- ggplot(df,aes(x=as.numeric(inter),y=gdpgrowth)) +  geom_point() + geom_smooth(method=lm) +
  ggtitle("GDP Growth vs. Better Data Quality") + theme(plot.title=element_text(hjust=0.5)) + xlab("inter")

graficooecd <- ggplot(df,aes(x=as.numeric(oecd),y=gdpgrowth)) +  geom_point() + geom_smooth(method=lm) +
  ggtitle("GDP Growth vs. oecd") + theme(plot.title=element_text(hjust=0.5)) + xlab("oecd")

ggarrange(graficooil,graficointer,graficooecd, labels = "AUTO")

##############################################################################################################################

lm( gdpgrowth ~ gdp60 + gdp85 + popgrowth + invest + school + literacy60 ,data=df[df$oil=='yes',] )
lm( gdpgrowth ~ gdp60 + gdp85 + popgrowth + invest + school + literacy60,data=df[df$oil=='no',] )

lm( gdpgrowth ~ gdp60 + gdp85 + popgrowth + invest + school + literacy60,data=df[df$inter=='yes',] )
lm( gdpgrowth ~ gdp60 + gdp85 + popgrowth + invest + school + literacy60,data=df[df$inter=='no',] )

lm( gdpgrowth ~ gdp60 + gdp85 + popgrowth + invest + school + literacy60,data=df[df$oecd=='yes',] )
lm( gdpgrowth ~ gdp60 + gdp85 + popgrowth + invest + school + literacy60,data=df[df$oecd=='no',] )


###############################################################################################################################
###########                                       Regression Models                                                 ###########
###############################################################################################################################


### First model with only numerical variables ###

Mod0res <- lm(gdpgrowth ~ gdp60 + gdp85 + popgrowth + invest + school + literacy60, data= df)
summary(Mod0res)
AIC(Mod0res)
BIC(Mod0res)

### Second Model adding the dummy variables ###

Mod1res <- update(Mod0res, ~ . + oecd + inter + oil)
summary(Mod1res)
AIC(Mod1res)
BIC(Mod1res)

### Third model removing variables that are not significant ###

Mod2res <- update(Mod1res, ~ . -school -oecd - oil)
summary(Mod2res)
AIC(Mod2res)
BIC(Mod2res)

### Forth model removing literacy60 ###

Mod3res <- update(Mod2res, ~ . -literacy60)
summary(Mod3res)
AIC(Mod3res)
BIC(Mod3res)

### Step model ###

Mod4res <- step(Mod2res)
summary(Mod4res)
AIC(Mod4res)
BIC(Mod4res)

###############################################################################################################################
###########                                           Outliers                                                      ###########
###############################################################################################################################


qqPlot(Mod4res) #47 e 35

outlierTest(Mod4res)
influenceIndexPlot(Mod4res)

influencePlot(Mod4res)
allinflmMod4 <- influence.measures(Mod4res)
summary(allinflmMod4)

print(df[47,])
print(df[43:53,])

### Removal of line 47 and creation of the fifth model ###

df_new <- df[-47,]
rownames(df_new) <- NULL ##reset index
Mod5res <-  lm(gdpgrowth ~ gdp60 + gdp85 + popgrowth + invest + inter + literacy60, 
               data=df_new) 

summary(Mod5res)
AIC(Mod5res) 
BIC(Mod5res)

### Sixth and final model creation ###

Mod6res <-  update(Mod5res, ~ . -literacy60)
summary(Mod6res)
AIC(Mod6res) 
BIC(Mod6res)


qqPlot(Mod6res) #35
outlierTest(Mod6res)
influenceIndexPlot(Mod6res)
influencePlot(Mod6res)

print(df_new[95,]) # Not an outlier
print(df_new[90:100,])

###############################################################################################################################
###########                                 Final regression model analysis                                         ###########
###############################################################################################################################

summary(Mod6res)
AIC(Mod6res) 
BIC(Mod6res)
anova(Mod6res)
confint(Mod6res)

###############################################################################################################################
###########                                       Independence of errors                                            ###########
###############################################################################################################################

plot(Mod6res, which = 1)  # Residuals vs. Fitted
plot(Mod6res, which = 2)  # Normal Q-Q plot

###############################################################################################################################
###########                          Heteroscedasticity and auto-correlation diagnostics                            ###########
###############################################################################################################################

########## Verifying ############

### Residuals vs Fitted Values ###

plot(Mod6res$fitted.values, resid(Mod6res), xlab = "Fitted Values", ylab = "Residuals", main = "Residuals vs. Fitted Values") 

### Breusch-Pagan Test ###
ols_plot_resid_fit(Mod6res)
ols_test_breusch_pagan(Mod6res)


########## Corrections ############

### Correction 1 utilizing Robust Standard Errors (RBS) ###

Mod6res$HCrobVCOV <- vcovHC(Mod6res)
robust_model <- coeftest(Mod6res,Mod6res$HCrobVCOV)

print(robust_model)


######### Autocorrelation

dwtest(Mod6res) # Not present

###############################################################################################################################
###########                                       Normality of Errors                                               ###########
###############################################################################################################################

residuals <- resid(Mod6res)
hist(residuals, main = "Histogram of Residuals")

qqnorm(residuals)
qqline(residuals)
shapiro.test(residuals)

###############################################################################################################################
###########                                    Multicolineariy Diagnostics                                          ###########
###############################################################################################################################

vif(Mod6res)

###############################################################################################################################
###########                                          Graphs                                                         ###########
###############################################################################################################################

grafico1 <- ggplot(data = df, aes(x = gdp60, y = gdpgrowth, colour= inter)) + geom_point() + facet_wrap(~inter, scales = "free")
grafico2 <- ggplot(data = df, aes(x = gdp60, y = gdpgrowth, colour= oil)) + geom_point() + facet_wrap(~oil, scales = "free")
grafico3 <- ggplot(data = df, aes(x = gdp60, y = gdpgrowth, colour= oecd)) + geom_point() + facet_wrap(~oecd, scales = "free")
grafico4 <- ggplot(data = df, aes(y = gdpgrowth, x = gdp60)) + geom_point() + geom_smooth(method=lm)
ggarrange(grafico1,grafico2,grafico3,grafico4, labels = "AUTO")

grafico5 <- ggplot(data = df, aes(x = gdp85, y = gdpgrowth, colour= inter)) + geom_point() + facet_wrap(~inter, scales = "free")
grafico6 <- ggplot(data = df, aes(x = gdp85, y = gdpgrowth, colour= oil)) + geom_point() + facet_wrap(~oil, scales = "free")
grafico7 <- ggplot(data = df, aes(x = gdp85, y = gdpgrowth, colour= oecd)) + geom_point() + facet_wrap(~oecd, scales = "free")
grafico8 <- ggplot(data = df, aes(y = gdpgrowth, x = gdp85)) + geom_point() + geom_smooth(method=lm)
ggarrange(grafico5,grafico6,grafico7,grafico8, labels = "AUTO")

grafico9 <- ggplot(data = df, aes(x = popgrowth, y = gdpgrowth, colour= inter)) + geom_point() + facet_wrap(~inter, scales = "free")
grafico10 <- ggplot(data = df, aes(x = popgrowth, y = gdpgrowth, colour= oil)) + geom_point() + facet_wrap(~oil, scales = "free")
grafico11 <- ggplot(data = df, aes(x = popgrowth, y = gdpgrowth, colour= oecd)) + geom_point() + facet_wrap(~oecd, scales = "free")
grafico12 <- ggplot(data = df, aes(y = gdpgrowth, x = popgrowth)) + geom_point() + geom_smooth(method=lm)
ggarrange(grafico9,grafico10,grafico11,grafico12, labels = "AUTO")

grafico13 <- ggplot(data = df, aes(x = invest, y = gdpgrowth, colour= inter)) + geom_point() + facet_wrap(~inter, scales = "free")
grafico14 <- ggplot(data = df, aes(x = invest, y = gdpgrowth, colour= oil)) + geom_point() + facet_wrap(~oil, scales = "free")
grafico15 <- ggplot(data = df, aes(x = invest, y = gdpgrowth, colour= oecd)) + geom_point() + facet_wrap(~oecd, scales = "free")
grafico16 <- ggplot(data = df, aes(y = gdpgrowth, x = invest)) + geom_point() + geom_smooth(method=lm)
ggarrange(grafico13,grafico14,grafico15,grafico16, labels = "AUTO")

grafico17 <- ggplot(data = df, aes(x = literacy60, y = gdpgrowth, colour= inter)) + geom_point() + facet_wrap(~inter, scales = "free")
grafico18 <- ggplot(data = df, aes(x = literacy60, y = gdpgrowth, colour= oil)) + geom_point() + facet_wrap(~oil, scales = "free")
grafico19 <- ggplot(data = df, aes(x = literacy60, y = gdpgrowth, colour= oecd)) + geom_point() + facet_wrap(~oecd, scales = "free")
grafico20 <- ggplot(data = df, aes(y = gdpgrowth, x = literacy60)) + geom_point() + geom_smooth(method=lm)
ggarrange(grafico17,grafico18,grafico19,grafico20, labels = "AUTO")

grafico21 <- ggplot(data = df, aes(x = school, y = gdpgrowth, colour= inter)) + geom_point() + facet_wrap(~inter, scales = "free")
grafico22 <- ggplot(data = df, aes(x = school, y = gdpgrowth, colour= oil)) + geom_point() + facet_wrap(~oil, scales = "free")
grafico23 <- ggplot(data = df, aes(x = school, y = gdpgrowth, colour= oecd)) + geom_point() + facet_wrap(~oecd, scales = "free")
grafico24 <- ggplot(data = df, aes(y = gdpgrowth, x = school)) + geom_point() + geom_smooth(method=lm)
ggarrange(grafico21,grafico22,grafico23,grafico24, labels = "AUTO")

### Histograms ###

ggplot(df, aes(x = gdp60)) +
  geom_histogram(fill = "skyblue", color = "black") +
  labs(title = "Histogram of GDP60", x = "GDP 60", y = "Frequency") +   theme_minimal()

ggplot(df, aes(x = gdp85)) +
  geom_histogram(fill = "skyblue", color = "black") +
  labs(title = "Histogram of GDP85", x = "GDP 85", y = "Frequency")+   theme_minimal()

ggplot(df, aes(x = popgrowth)) +
  geom_histogram(fill = "skyblue", color = "black") +
  labs(title = "Histogram of Popgrowth",x = "Population Growth", y = "Frequency")+   theme_minimal()

ggplot(df, aes(x = invest)) +
  geom_histogram(fill = "skyblue", color = "black") +
  labs(title = "Histogram of Invest", x = "Invest", y = "Frequency")+   theme_minimal()

ggplot(df, aes(x = school)) +
  geom_histogram(fill = "skyblue", color = "black") +
  labs(title = "Histogram of School", x = "School", y = "Frequency")+   theme_minimal()

ggplot(df, aes(x = literacy60)) +
  geom_histogram(fill = "skyblue", color = "black") +
  labs(title = "Histogram of Literacy60", x = "Literacy60", y = "Frequency")+   theme_minimal()

ggplot(df, aes(x = gdpgrowth)) +
  geom_histogram(fill = "skyblue", color = "black") +
  labs(title = "Histogram of GDPgrowth", x = "GDPgrowth", y = "Frequency")+   theme_minimal()

############################################################################################################################

ggplot(df, aes(x = gdp60)) +
  geom_density(fill = "skyblue", color = "gray", alpha = 0.7) +
  geom_vline(aes(xintercept = mean(gdp60)), color = "red", linetype = "dashed", size = 1) +
  geom_vline(aes(xintercept = quantile(gdp60, 0.75)), color = "blue", linetype = "dashed", size = 1) +
  geom_vline(aes(xintercept = quantile(gdp60, 0.25)), color = "blue", linetype = "dashed", size = 1) +
  geom_vline(aes(xintercept = median(gdp60)), color = "black", linetype = "dashed", size = 1) +
  labs(
    title = paste("Density Plot GDP60 with Kurtosis", round(kurtosis(df$gdp60), 2)),
    x = "GDP60",
    y = "Density"
  ) +
  annotate("text", x = mean(df$gdp60), y = 0.00025, label = "Mean", color = "red", angle = 90, vjust = -0.5) +
  annotate("text", x = median(df$gdp60), y = 0.00025, label = "Median", color = "black", angle = 90, vjust = -0.5) +
  annotate("text", x = quantile(df$gdp60, 0.75), y = 0.00025, label = "Q3", color = "blue", angle = 90, vjust = -0.5) +
  annotate("text", x = quantile(df$gdp60, 0.25), y = 0.00025, label = "Q1", color = "blue", angle = 90, vjust = -0.5) +
  theme_minimal() +
  scale_color_manual(
    name = "Legend",
    values = c("Mean" = "red", "Median" = "black", "Q3" = "blue", "Q1" = "blue"),
    labels = c("Mean", "Median", "Q3", "Q1")
  ) +
  geom_point(aes(x = -Inf, y = -Inf, color = "Mean")) +
  geom_point(aes(x = -Inf, y = -Inf, color = "Median")) +
  geom_point(aes(x = -Inf, y = -Inf, color = "Q3")) +
  geom_point(aes(x = -Inf, y = -Inf, color = "Q1"))
 
ggplot(df, aes(x = gdp85)) +
  geom_density(fill = "skyblue", color = "gray", alpha = 0.7) +
  geom_vline(aes(xintercept = mean(gdp85)), color = "red", linetype = "dashed", size = 1) +
  geom_vline(aes(xintercept = quantile(gdp85, 0.75)), color = "blue", linetype = "dashed", size = 1) +
  geom_vline(aes(xintercept = quantile(gdp85, 0.25)), color = "blue", linetype = "dashed", size = 1) +
  geom_vline(aes(xintercept = median(gdp85)), color = "black", linetype = "dashed", size = 1) +
  labs(
    title = paste("Density Plot GDP85 with Kurtosis", round(kurtosis(df$gdp85), 2)),
    x = "GDP85",
    y = "Density"
  ) +
  annotate("text", x = mean(df$gdp85), y = 0.00015, label = "Mean", color = "red", angle = 90, vjust = -0.5) +
  annotate("text", x = median(df$gdp85), y = 0.00015, label = "Median", color = "black", angle = 90, vjust = -0.5) +
  annotate("text", x = quantile(df$gdp85, 0.75), y = 0.00015, label = "Q3", color = "blue", angle = 90, vjust = -0.5) +
  annotate("text", x = quantile(df$gdp85, 0.25), y = 0.00015, label = "Q1", color = "blue", angle = 90, vjust = -0.5) +
  theme_minimal() +
  scale_color_manual(
    name = "Legend",
    values = c("Mean" = "red", "Median" = "black", "Q3" = "blue", "Q1" = "blue"),
    labels = c("Mean", "Median", "Q3", "Q1")
  ) +
  geom_point(aes(x = -Inf, y = -Inf, color = "Mean")) +
  geom_point(aes(x = -Inf, y = -Inf, color = "Median")) +
  geom_point(aes(x = -Inf, y = -Inf, color = "Q3")) +
  geom_point(aes(x = -Inf, y = -Inf, color = "Q1"))

ggplot(df, aes(x = invest)) +
  geom_density(fill = "skyblue", color = "gray", alpha = 0.7) +
  geom_vline(aes(xintercept = mean(invest)), color = "red", linetype = "dashed", size = 1) +
  geom_vline(aes(xintercept = quantile(invest, 0.75)), color = "blue", linetype = "dashed", size = 1) +
  geom_vline(aes(xintercept = quantile(invest, 0.25)), color = "blue", linetype = "dashed", size = 1) +
  geom_vline(aes(xintercept = median(invest)), color = "black", linetype = "dashed", size = 1) +
  labs(
    title = paste("Density Plot Invest with Kurtosis", round(kurtosis(df$invest), 2)),
    x = "Invest",
    y = "Density"
  ) +
  annotate("text", x = mean(df$invest), y = 0.05, label = "Mean", color = "red", angle = 90, vjust = -0.5) +
  annotate("text", x = median(df$invest), y = 0.05, label = "Median", color = "black", angle = 90, vjust = -0.5) +
  annotate("text", x = quantile(df$invest, 0.75), y = 0.05, label = "Q3", color = "blue", angle = 90, vjust = -0.5) +
  annotate("text", x = quantile(df$invest, 0.25), y = 0.05, label = "Q1", color = "blue", angle = 90, vjust = -0.5) +
  theme_minimal() +
  scale_color_manual(
    name = "Legend",
    values = c("Mean" = "red", "Median" = "black", "Q3" = "blue", "Q1" = "blue"),
    labels = c("Mean", "Median", "Q3", "Q1")
  ) +
  geom_point(aes(x = -Inf, y = -Inf, color = "Mean")) +
  geom_point(aes(x = -Inf, y = -Inf, color = "Median")) +
  geom_point(aes(x = -Inf, y = -Inf, color = "Q3")) +
  geom_point(aes(x = -Inf, y = -Inf, color = "Q1"))

ggplot(df, aes(x = popgrowth)) +
  geom_density(fill = "skyblue", color = "gray", alpha = 0.7) +
  geom_vline(aes(xintercept = mean(popgrowth)), color = "red", linetype = "dashed", size = 1) +
  geom_vline(aes(xintercept = quantile(popgrowth, 0.75)), color = "blue", linetype = "dashed", size = 1) +
  geom_vline(aes(xintercept = quantile(popgrowth, 0.25)), color = "blue", linetype = "dashed", size = 1) +
  geom_vline(aes(xintercept = median(popgrowth)), color = "black", linetype = "dashed", size = 1) +
  labs(
    title = paste("Density Plot Popgrowth with Kurtosis", round(kurtosis(df$popgrowth), 2)),
    x = "Popgrowth",
    y = "Density"
  ) +
  annotate("text", x = mean(df$popgrowth), y = 0.55, label = "Mean", color = "red", angle = 90, vjust = -0.5) +
  annotate("text", x = median(df$popgrowth), y = 0.55, label = "Median", color = "black", angle = 90, vjust = -0.5) +
  annotate("text", x = quantile(df$popgrowth, 0.75), y = 0.55, label = "Q3", color = "blue", angle = 90, vjust = -0.5) +
  annotate("text", x = quantile(df$popgrowth, 0.25), y = 0.55, label = "Q1", color = "blue", angle = 90, vjust = -0.5) +
  theme_minimal() +
  scale_color_manual(
    name = "Legend",
    values = c("Mean" = "red", "Median" = "black", "Q3" = "blue", "Q1" = "blue"),
    labels = c("Mean", "Median", "Q3", "Q1")
  ) +
  geom_point(aes(x = -Inf, y = -Inf, color = "Mean")) +
  geom_point(aes(x = -Inf, y = -Inf, color = "Median")) +
  geom_point(aes(x = -Inf, y = -Inf, color = "Q3")) +
  geom_point(aes(x = -Inf, y = -Inf, color = "Q1"))


ggplot(df, aes(x = gdpgrowth)) +
  geom_density(fill = "skyblue", color = "gray", alpha = 0.7) +
  geom_vline(aes(xintercept = mean(gdpgrowth)), color = "red", linetype = "dashed", size = 1) +
  geom_vline(aes(xintercept = quantile(gdpgrowth, 0.75)), color = "blue", linetype = "dashed", size = 1) +
  geom_vline(aes(xintercept = quantile(gdpgrowth, 0.25)), color = "blue", linetype = "dashed", size = 1) +
  geom_vline(aes(xintercept = median(gdpgrowth)), color = "black", linetype = "dashed", size = 1) +
  labs(
    title = paste("Density Plot GDPgrowth with Kurtosis", round(kurtosis(df$gdpgrowth), 2)),
    x = "GDPgrowth",
    y = "Density"
  ) +
  annotate("text", x = mean(df$gdpgrowth), y = 0.3, label = "Mean", color = "red", angle = 90, vjust = -0.5) +
  annotate("text", x = median(df$gdpgrowth), y = 0.3, label = "Median", color = "black", angle = 90, vjust = -0.5) +
  annotate("text", x = quantile(df$gdpgrowth, 0.75), y = 0.3, label = "Q3", color = "blue", angle = 90, vjust = -0.5) +
  annotate("text", x = quantile(df$gdpgrowth, 0.25), y = 0.3, label = "Q1", color = "blue", angle = 90, vjust = -0.5) +
  theme_minimal() +
  scale_color_manual(
    name = "Legend",
    values = c("Mean" = "red", "Median" = "black", "Q3" = "blue", "Q1" = "blue"),
    labels = c("Mean", "Median", "Q3", "Q1")
  ) +
  geom_point(aes(x = -Inf, y = -Inf, color = "Mean")) +
  geom_point(aes(x = -Inf, y = -Inf, color = "Median")) +
  geom_point(aes(x = -Inf, y = -Inf, color = "Q3")) +
  geom_point(aes(x = -Inf, y = -Inf, color = "Q1"))


ggplot(df, aes(x = school)) +
  geom_density(fill = "skyblue", color = "gray", alpha = 0.7) +
  geom_vline(aes(xintercept = mean(school)), color = "red", linetype = "dashed", size = 1) +
  geom_vline(aes(xintercept = quantile(school, 0.75)), color = "blue", linetype = "dashed", size = 1) +
  geom_vline(aes(xintercept = quantile(school, 0.25)), color = "blue", linetype = "dashed", size = 1) +
  geom_vline(aes(xintercept = median(school)), color = "black", linetype = "dashed", size = 1) +
  labs(
    title = paste("Density Plot School with Kurtosis", round(kurtosis(df$school), 2)),
    x = "School",
    y = "Density"
  ) +
  annotate("text", x = mean(df$school), y = 0.15, label = "Mean", color = "red", angle = 90, vjust = -0.5) +
  annotate("text", x = median(df$school), y = 0.15, label = "Median", color = "black", angle = 90, vjust = -0.5) +
  annotate("text", x = quantile(df$school, 0.75), y = 0.15, label = "Q3", color = "blue", angle = 90, vjust = -0.5) +
  annotate("text", x = quantile(df$school, 0.25), y = 0.15, label = "Q1", color = "blue", angle = 90, vjust = -0.5) +
  theme_minimal() +
  scale_color_manual(
    name = "Legend",
    values = c("Mean" = "red", "Median" = "black", "Q3" = "blue", "Q1" = "blue"),
    labels = c("Mean", "Median", "Q3", "Q1")
  ) +
  geom_point(aes(x = -Inf, y = -Inf, color = "Mean")) +
  geom_point(aes(x = -Inf, y = -Inf, color = "Median")) +
  geom_point(aes(x = -Inf, y = -Inf, color = "Q3")) +
  geom_point(aes(x = -Inf, y = -Inf, color = "Q1"))



ggplot(df, aes(x = literacy60)) +
  geom_density(fill = "skyblue", color = "gray", alpha = 0.7) +
  geom_vline(aes(xintercept = mean(literacy60)), color = "red", linetype = "dashed", size = 1) +
  geom_vline(aes(xintercept = quantile(literacy60, 0.75)), color = "blue", linetype = "dashed", size = 1) +
  geom_vline(aes(xintercept = quantile(literacy60, 0.25)), color = "blue", linetype = "dashed", size = 1) +
  geom_vline(aes(xintercept = median(literacy60)), color = "black", linetype = "dashed", size = 1) +
  labs(
    title = paste("Density Plot Literacy60 with Kurtosis", round(kurtosis(df$literacy60), 2)),
    x = "Literacy60",
    y = "Density"
  ) +
  annotate("text", x = mean(df$literacy60), y = 0.015, label = "Mean", color = "red", angle = 90, vjust = -0.5) +
  annotate("text", x = median(df$literacy60), y = 0.015, label = "Median", color = "black", angle = 90, vjust = -0.5) +
  annotate("text", x = quantile(df$literacy60, 0.75), y = 0.015, label = "Q3", color = "blue", angle = 90, vjust = -0.5) +
  annotate("text", x = quantile(df$literacy60, 0.25), y = 0.015, label = "Q1", color = "blue", angle = 90, vjust = -0.5) +
  theme_minimal() +
  scale_color_manual(
    name = "Legend",
    values = c("Mean" = "red", "Median" = "black", "Q3" = "blue", "Q1" = "blue"),
    labels = c("Mean", "Median", "Q3", "Q1")
  ) +
  geom_point(aes(x = -Inf, y = -Inf, color = "Mean")) +
  geom_point(aes(x = -Inf, y = -Inf, color = "Median")) +
  geom_point(aes(x = -Inf, y = -Inf, color = "Q3")) +
  geom_point(aes(x = -Inf, y = -Inf, color = "Q1"))
