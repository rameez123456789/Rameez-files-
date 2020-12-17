
#reading file and naming them 
library(readr)
ad=read_csv("C:/Users/Dell/Desktop/Automobile_data.csv")
View(ad)
attach(ad)
summary(ad)

rowSums(is.na(ad)) 
colSums(is.na(ad))# count number of NA

#Not required
#ad[ad$`normalized-losses`=="?","norma"] <- NA 
#ad$`normalized-losses`[ad$`normalized-losses` == "?"] <- NA #relplace ? with NA
#table(ad$`normalized-losses`[ad$`normalized-losses`=="?"])#relacement
#ad$price[ad$price == "?"] <- NA#replacement 
#ad$`num-of-doors`[ad$`num-of-doors` == "?"] <- NA#replacement 
#ad$bore[ad$bore == "?"] <- NA #replacement 
#ad$stroke[ad$stroke == "?"] <- NA #replacement 
#ad$`peak-rpm`[ad$`peak-rpm` == "?"] <- NA #replacement 
#ad$stroke[ad$stroke == "?"] <- NA #replacement 
#ad$horsepower[ad$horsepower == "?"] <- NA #replacement 
#ad$`peak-rpm`[ad$`peak-rpm` == "?"] <- NA #replacement 

#colown tht have ? in them
count <- function(x, n){ length((which(x == n))) }#counting function of ?
count(make,"?")#using fun
count(`num-of-doors`,"?")#number of doors 
#BORE
count(bore,"?")
#Stroke
count(stroke,"?")
#HorsePower
count(horsepower,"?")
#Peakrpm
count(`peak-rpm`,"?")
#price
count(ad$price,"?")
#normalizeLOsses
count(ad$`normalized-losses`,"?")


##replacement ? 
##Find ? then replace with NA and 
##convert object into nummeric and replace NA with mean
##Normalized Losses
ifelse(ad$`normalized-losses`=="?",NA,ad$`normalized-losses`)
ad$`normalized-losses` <- as.numeric(as.character(ad$`normalized-losses`))
ad$`normalized-losses`[is.na(ad$`normalized-losses`)]=mean(ad$`normalized-losses`,na.rm=TRUE)


#Horsepower
ifelse(ad$horsepower=="?",NA,ad$horsepower)
ad$horsepower = as.numeric(as.character(ad$horsepower))
ad$horsepower[is.na(ad$horsepower)]=mean(ad$horsepower,na.rm=TRUE)

#Peakrpm
ifelse(ad$`peak-rpm`=="?",NA,ad$`peak-rpm`)
ad$`peak-rpm` =as.numeric(as.character(ad$`peak-rpm`))
ad$`peak-rpm`[is.na(ad$`peak-rpm`)]=mean(ad$`peak-rpm`,na.rm=TRUE)

#BORE
ifelse(ad$bore=="?",NA,ad$bore)
ad$bore = as.numeric(as.character(ad$bore))
ad$bore[is.na(ad$bore)]=mean(ad$bore,na.rm=TRUE)

#Stroke
ifelse(ad$stroke=="?",NA,ad$stroke)
ad$stroke = as.numeric(as.character(ad$stroke))
ad$stroke[is.na(ad$stroke)]=mean(ad$stroke,na.rm=TRUE)

#Price
ifelse(ad$price=="?",NA,ad$price)
ad$price = as.numeric(as.character(ad$price))
ad$price[is.na(ad$price)]=mean(ad$price,na.rm=TRUE)
summary(ad$price)


#Number of Doors
table(ad$`num-of-doors`)
ad$`num-of-doors`[ad$`num-of-doors`=='?']='four'

#finding Types of Fuel
Fuel <- ad$`fuel-type`
hist(Fuel)
table(`fuel-type`)

#feature engineering of catergorical values 
#Make Binning
table(ad$make)
ad$car_Jap = ifelse(ad$make == "honda"| ad$make == "isuzu"|ad$make == "mazda"|ad$make == "mitsubishi"|ad$make == "nissan"|ad$make == "subaru"|ad$make == "toyota",1,0)
ad$car_Euro = ifelse(ad$make == "alfa-romero"| ad$make == "audi"|ad$make == "bmw"|ad$make == "jaguar"|ad$make == "mercedes-benz"|ad$make == "peugot"|ad$make == "porsche"|ad$make == "renault"|ad$make == "peugot"|ad$make == "saab"|ad$make == "volkswagen"|ad$make == "volvo",1,0)
ad$car_Amer = ifelse(ad$make == "chevrolet"| ad$make == "dodge"|ad$make == "mercury"|ad$make == "plymouth",1,0)
#Number of doors
table(ad$`num-of-doors`)
ad$`num-of-doors` = ifelse(ad$`num-of-doors` == "four",1,0)
#Fuel Type
table(ad$`fuel-type`)
ad$`fuel-type` = ifelse(ad$`fuel-type` == "gas",1,0)
#Aspiration
table(ad$aspiration)
ad$aspiration = ifelse(ad$aspiration == "std",1,0)
#BodyStyle 
table(ad$`body-style`)
ad$`body-style`= ifelse(ad$`body-style` == "sedan"| ad$`body-style` == "wagon",1,0)
#DRive Wheel
table(ad$`drive-wheels`)
ad$`drive-wheels`= ifelse(ad$`drive-wheels` == "rwd"| ad$`drive-wheels` == "4wd",0,1)
table(ad$`engine-location`)
#engine Location
ad$`engine-location`= ifelse(ad$`engine-location` == "front",1,0)
#engine type
table(ad$`engine-type`)
ad$`engine-type`= ifelse(ad$`engine-type` == "ohc",1,0)
#Number of Cylinders
table(ad$`num-of-cylinders`)
ad$`num-of-cylinders`= ifelse(ad$`num-of-cylinders` == "four",1,0)
#Fuel system
table(ad$`fuel-system`)
ad$`fuel-system`= ifelse(ad$`fuel-system` == "mpfi",1,0)
#DRoping Make from dataframe
ad <- ad[ -c(3) ]
summary(ad)



#catergorize the variables mentioned below
#symboling
ad$symboling=as.factor(ad$symboling)
table(ad$`fuel-type`)
summary(ad)
attach(ad)
#fuel type
ad$`fuel-type`=as.factor(ad$`fuel-type`)
levels(`fuel-type`)
table(`fuel-type`)
#aspiration
ad$aspiration=as.factor(ad$aspiration)
levels(aspiration)
table(aspiration)
#Number of Doors 
ad$`num-of-doors`=as.factor(ad$`num-of-doors`)
#Body Style 
ad$`body-style`=as.factor(ad$`body-style`)
#DRive wheels 
ad$`drive-wheels`=as.factor(ad$`drive-wheels`)
ad$`engine-location`=as.factor(ad$`engine-location`)
ad$`engine-type`=as.factor(ad$`engine-type`)
ad$`num-of-cylinders`=as.factor(ad$`num-of-cylinders`)
ad$`fuel-system`=as.factor(ad$`fuel-system`)
ad$make=as.factor(ad$make)
levels(make)
table(make)
summary(ad)

#Feature selection using random forest all features model
#install.packages('randomForest')
library(randomForest)
myforest=randomForest(price~symboling+
                        `normalized-losses`+
                         car_Amer+car_Euro+car_Jap+
                        `fuel-system`+
                        `fuel-type`+
                         aspiration+
                        `num-of-doors`+
                         `body-style`+
                        `drive-wheels`+
                        `engine-location`+
                        `engine-size`+
                        `engine-type`+
                        `wheel-base`+
                         length+
                         bore+
                         stroke+
                        `compression-ratio`+
                         horsepower+
                        `peak-rpm`+
                        `city-mpg`+
                        `highway-mpg`+
                         width+
                         height+
                        `num-of-cylinders`+
                        `curb-weight`, ntree=800, data=ad, importance=TRUE, na.action = na.omit)
myforest
imp = importance(myforest)
imp
varImpPlot(myforest)#ploting features and importance chart

#org error out-of-bag performance of our model #not required
myforest=randomForest(price~symboling+
                        `normalized-losses`+
                        make+
                        `fuel-system`+
                        `fuel-type`+
                        aspiration+
                        `num-of-doors`+
                        `body-style`+
                        `drive-wheels`+
                        `engine-location`+
                        `engine-size`+
                        `engine-type`+
                        `wheel-base`+
                        length+
                        bore+
                        stroke+
                        `compression-ratio`+
                        horsepower+
                        `peak-rpm`+
                        `city-mpg`+
                        `highway-mpg`+
                        width+
                        height+
                        `num-of-cylinders`+
                        `curb-weight`, ntree=800, data=ad, importance=TRUE, na.action = na.omit,do.trace=50)

# correlation for all variables
round(cor(ad),
      digits = 2 # rounded to 2 decimals
)
###PCA model
>install.packages("ggplot2")
>install.packages("ggfortify")
library(ggplot2)
library(ggfortify)
ad_labels=ad[,c(3,4,5,6,7,8,9,10,11)]

#ad_vars=ad[,c(1,2,9,10,11,12,13,14,16,17,19,20,21,22,23,24,25)]#continous variables 
ad_vars=ad[,c(1:28)]# All variables 
ad_vars
ad_labels

# modelPCA using prcomp
#library(ggplot2)
#library(GGally)
#ggpairs(ad_vars)
pca=prcomp(ad_vars, scale=TRUE)
pca

#ploting two PCA components 
autoplot(pca, data = ad_vars, loadings = TRUE, loadings.label = TRUE )

#Indentification of various features and location of those cars on the map
autoplot(pca, data = ad_vars, loadings = TRUE,
col=ifelse(ad_labels$`fuel-type`==0,'blue','transparent'), loadings.label = TRUE )


#ploting corelation vizvalization
correlations <- cor(ad,method='pearson')
cormat<-signif(cor(ad),2)
cormat
# Get some colors
col<- colorRampPalette(c("blue", "white", "red"))(20)
heatmap(cormat, col=col, symm=TRUE)


#To find the percentage of variance in PCA
pve=(pca$sdev^2)/sum(pca$sdev^2)
par(mfrow=c(1,2))
plot(pve, ylim=c(0,1))
plot(cumsum(pve), ylim=c(0,1))


# REgression to find price using BOOSTING method 
#with all varibles 
library(gbm)
set.seed (1)
boosted=gbm(price~symboling+
                        `normalized-losses`+
                         car_Amer+car_Euro+car_Jap+
                        `fuel-system`+
                        `fuel-type`+
                         aspiration+
                        `num-of-doors`+
                         `body-style`+
                        `drive-wheels`+
                        `engine-location`+
                        `engine-size`+
                        `engine-type`+
                        `wheel-base`+
                         length+
                         bore+
                         stroke+
                        `compression-ratio`+
                         horsepower+
                        `peak-rpm`+
                        `city-mpg`+
                        `highway-mpg`+
                         width+
                         height+
                        `num-of-cylinders`+
                        `curb-weight`,data=ad,distribution='gaussian',n.trees=10000,interaction.depth=4) 
summary(boosted) 
predicted_score=predict(boosted, newdata=ad, n.trees=10000)#predicted values 
mean((predicted_score -ad$price)^2) #mean of predicted values 

#revised after feature selection
library(gbm)
set.seed (1)
boosted=gbm(ad$price~symboling+
                     `engine-size`+
                      `normalized-losses`+
                       car_Euro+
                      `fuel-system`+
                      `body-style`+
                      `wheel-base`+
                       stroke+
                       horsepower+
                       `peak-rpm`+
                      `city-mpg`+
                       `highway-mpg`+
                       width+height+
                       `curb-weight`,data=ad,distribution='gaussian',n.trees=10000,interaction.depth=4) 
summary(boosted) #guassian is for regression
predicted_score=predict(boosted, newdata=ad, n.trees=10000)
mean((predicted_score -ad$price)^2) #58294.14answer 


