setwd("P:\\Spring 2020\\ML\\Final project")

#Loading all the required packages
library(caret)
library(tidyverse)
library(arsenal)
library(GGally)
library(party)
library(stringr)
library(dplyr)
library(ggplot2)
library(randomForest)
library(corrplot)

#Reading the file and inserting it into a variable
covid_19 = read.csv('COVID19_line_list_data.csv')
View(covid_19)
summary(covid_19)


## Extract the main symptoms of COVID-19
covid_19$mainsymptoms = str_detect(covid_19$symptom, 'cough|fever|throat|breathlessness|dyspnea|pneumonia|malaise')

#Creating a subfile with only the essential variables
covid_19 = covid_19 %>% select(recovered,reporting.date, country, gender, age, visitingWuhan, fromWuhan, mainsymptoms, symptom)  %>% 
  mutate(
    recovered = ifelse(recovered == '0', 0, 1),
    country = factor(country),
    gender = factor(gender),
    recovered = factor(recovered, label = c('0','1')),
    reporting.date = as.Date(reporting.date, format = c('$d/$m/$Y')),
    mainsymptons = factor(mainsymptoms)
  )



#Plotting Age vs Recoveries
ggplot(covid_19, aes(recovered, age, fill = recovered))+
  geom_boxplot()+
  theme_classic()+
  scale_fill_manual(values = c('blue','red'))

#Plotting Gender vs Recoveries
ggplot(covid_19, aes(recovered, fill = gender))+
  geom_bar(position ='fill')+
  theme_classic()+
  scale_fill_manual(values = c('red','blue','black'))

#Most recoveries were in young patients
table1 = tableby(recovered ~ age + gender, total = FALSE, data = covid_19)
summary(table1, text = TRUE)



#Creating a dataset with only the essential columns used to create the logistic model
covidLR = covid_19 %>% select(recovered, gender, age,  visitingWuhan, fromWuhan, mainsymptoms)


#Creating a train and a test set in the ratio 75:25 respectively
Train = covidLR[1:577,1:6]
Test = covidLR[578:825,1:6]


#Building the logistic regression model
log_model = glm(formula = recovered~., family = binomial(link = "logit"), data= Train)
summary(log_model)

#Predicting the recoveries for the Test dataset and comparing it with the Test dataset to determine accuracy of the logistic model
result = predict(log_model, newdata=Test[,2:6], type="response")
result
result = ifelse(result>0.5,1,0) 
result
accuracy = mean(result == Test$recovered)
accuracy


#Two models training/testing/comparison
num_iterations = 500
acc_history = list(num_iterations)
acc_history_visitingwuhan = list(num_iterations)


for (i in 1:num_iterations) {
  inTrain = createDataPartition(y=covidLR$recovered, p=0.75, list=FALSE)
  X_train = covidLR[inTrain, ]
  X_test = covidLR[-inTrain, ]
  model = glm(formula = recovered~., family = binomial(link = "logit"), data= X_train)
  result = predict(model, newdata=X_test[,2:6], type="response")
  result = ifelse(result>0.5,1,0) 
  accuracy = mean(result == X_test$recovered)
  acc_history[[i]] = accuracy
  model_1 = glm(formula = recovered~visitingWuhan, family = binomial(link = "logit"), data= X_train)
  result_visitingwuhan = predict(model_1, newdata=X_test[,2:6], type="response")
  result_visitingwuhan = ifelse(result_visitingwuhan>0.5,1,0) 
  accuracy_visitingwuhan = mean(result_visitingwuhan == X_test$recovered)
  acc_history_visitingwuhan[[i]] = accuracy_visitingwuhan
}


##Printing average accuracy for 500 iterations
sum_acc = 0
for (i in 1:num_iterations) {
  sum_acc = sum_acc + acc_history[[i]]
}
ave_acc = sum_acc/num_iterations
print(ave_acc)

sum_acc_1 = 0
for (i in 1:num_iterations) {
  sum_acc_1 = sum_acc_1 + acc_history_visitingwuhan[[i]]
}
ave_acc_1 = sum_acc_1/num_iterations
print(ave_acc_1)

#Welch test to comare the 2 models
df1 = data.frame(matrix(unlist(acc_history), nrow=length(acc_history), byrow=T))
df1$group = 1
df1$i = seq.int(nrow(df1))
names(df1)[1] = "accuracy"
df2 = data.frame(matrix(unlist(acc_history_visitingwuhan), nrow=length(acc_history_visitingwuhan), byrow=T))
df2$group = 0
df2$i = seq.int(nrow(df2))
names(df2)[1] = "accuracy"
df3 = rbind(df1, df2)
ggplot(data = df3, aes(x=i, y=accuracy, color=factor(group))) + xlab("round of test") +
  ylab("accuracy score") + geom_point() + labs(color="group")
t.test(accuracy~group, data=df3)



############################## Random forest ###################################
#Creating a dataset with only the essential columns used to create the RandomForest model
covid_RF = covid_19 %>% select(recovered, gender, age,  visitingWuhan, fromWuhan, mainsymptoms)

#Creating a train and a test set in the ratio 75:25 respectively
Train_RF = covid_RF[1:577,1:6]
Test_RF = covid_RF[578:825,1:6]


##Building the RandomForest model
model_RF = randomForest(recovered~., data=Train_RF)
model_RF

#Predicting the recoveries for the Test dataset and comparing it with the Test dataset to determine accuracy of the RandomForest model
prediction_RF = predict(model_RF, newdata=Test_RF)
prediction_RF
accuracy_RF = mean(prediction_RF == Test_RF$recovered)
accuracy_RF

#Two models training/testing/comparison
num_iterations_RF = 50
acc_history_RF = list(num_iterations_RF)
acc_history_1_RF = list(num_iterations_RF)


for (i in 1:num_iterations_RF) {
  inTrain_RF = createDataPartition(y=covid_RF$recovered, p=0.75, list=FALSE)
  X_train_RF = covid_RF[inTrain_RF, ]
  X_test_RF = covid_RF[-inTrain_RF, ]
  model_RF = randomForest(recovered~., data=X_train_RF)
  model1_RF = randomForest(recovered~visitingWuhan, data=X_train_RF)
  prediction_RF = predict(model_RF, newdata=X_test_RF)
  prediction1_RF = predict(model1_RF, newdata=X_test_RF)
  accuracy_RF = mean(prediction_RF == X_test_RF$recovered)
  accuracy1_RF = mean(prediction1_RF == X_test_RF$recovered)
  acc_history_RF[[i]] = accuracy_RF
  acc_history_1_RF[[i]] = accuracy1_RF
}
for (i in 1:num_iterations_RF) {
  print(acc_history_RF[[i]])
}
sum_acc_RF = 0
for (i in 1:num_iterations_RF) {
  sum_acc_RF = sum_acc_RF + acc_history_RF[[i]]
}

##print average accuracy for 50 iterations
ave_acc_RF = sum_acc_RF/num_iterations_RF
print(ave_acc_RF)

sum_acc_RF_1 = 0
for (i in 1:num_iterations_RF) {
  sum_acc_RF_1 = sum_acc_RF_1 + acc_history_1_RF[[i]]
}
ave_acc_RF_1 = sum_acc_RF_1/num_iterations_RF
print(ave_acc_RF_1)


#Welch test to comare the 2 models
df1_RF = data.frame(matrix(unlist(acc_history_RF), nrow=length(acc_history_RF), byrow=T))
df1_RF$group = 1
df1_RF$i = seq.int(nrow(df1_RF))
names(df1_RF)[1] = "accuracy_RF"
df2_RF = data.frame(matrix(unlist(acc_history_1_RF), nrow=length(acc_history_1_RF), byrow=T))
df2_RF$group = 0
df2_RF$i = seq.int(nrow(df2_RF))
names(df2_RF)[1] = "accuracy_RF"

df3_RF = rbind(df1_RF,df2_RF)
ggplot(data = df3_RF, aes(x=i, y=accuracy_RF, color=factor(group))) + xlab("round of test") +
  ylab("accuracy score") + geom_point() + labs(color="group")
t.test(accuracy_RF~group, data=df3_RF)
View(covidLR)
View(covid_RF)