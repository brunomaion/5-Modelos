#RF


mseRF = []
rmseRF= []
maeRF= []

p_RFcriterion = []
p_RFn_estimators  = []
p_RFmin_samples_leaf = []

param_grid = {
    'criterion': [(100,), (200,), (300,), (400,), (500,)],
    'n_estimators': ['relu', 'logistic', 'tanh', 'identity'],
    'min_samples_leaf': [400,600,800,1000],
}
melhor_validacao_RF = 1000
best_paramsRF = {}



for i in range(20):
    print(i+1)
    for criterion in param_grid['criterion']:
        for n_estimators in param_grid['n_estimators']:
            for min_samples_leaf in param_grid['min_samples_leaf']:
                RF = RandomForestRegressor(
                    criterion=criterion,
                    n_estimators=n_estimators,
                    min_samples_leaf=min_samples_leaf,
                )
                RF.fit(x_treino, y_treino)
                opiniao = RF.predict(x_teste)
                mse_validacao_RF = mean_squared_error(y_teste, opiniao)
                
                if (mse_validacao_RF < melhor_validacao_RF):
                    melhor_validacao_RF = mse_validacao_RF
                    best_paramsRF = {
                        'criterion': criterion,
                        'n_estimators': n_estimators,
                        'min_samples_leaf': min_samples_leaf,
                        'learning_rate_init': learning_rate_init
                    }


    print(melhor_validacao_RF)
    print(mse_validacao_RF)
    print(best_paramsRF)

    RF = RFRegressor(
                    criterion=best_paramsRF['criterion'],
                    n_estimators=best_paramsRF['n_estimators'],
                    solver='adam',
                    min_samples_leaf=best_paramsRF['min_samples_leaf'],
                    learning_rate_init=best_paramsRF['learning_rate_init']
                )
    
    RF.fit(x_treino, y_treino)
    opiniao_RF = RF.predict(x_teste)

    mae = mean_absolute_error(y_teste, opiniao_RF).round(2)
    mse = mean_squared_error(y_teste, opiniao_RF).round(2)
    rmse = np.sqrt(mse).round(2)

    maeRF.append(mae)
    mseRF.append(mse)
    rmseRF.append(rmse)

    p_RFcriterion.append(best_paramsRF['criterion'])
    p_RFn_estimators.append(best_paramsRF['n_estimators'])
    p_RFmin_samples_leaf.append(best_paramsRF['min_samples_leaf'])
    p_RFLearningRate.append(best_paramsRF['learning_rate_init'])

# RESULTADOS !!!!
maxp_RFcriterion = max(set(p_RFcriterion), key=p_RFcriterion.count)
maxp_RFn_estimators = max(set(p_RFn_estimators), key=p_RFn_estimators.count)
maxp_RFmin_samples_leaf = max(set(p_RFmin_samples_leaf), key=p_RFmin_samples_leaf.count)
maxp_RFLearningRate = max(set(p_RFLearningRate), key=p_RFLearningRate.count)


media_maeRF = np.mean(maeRF)
media_mseRF = np.mean(mseRF)
media_rmseRF = np.mean(rmseRF)

p_RFcriterion.append(maxp_RFcriterion)
p_RFn_estimators.append(p_RFn_estimators)
p_RFmin_samples_leaf.append(p_RFmin_samples_leaf)
p_RFLearningRate.append(p_RFLearningRate)

maeRF.append(media_maeRF)
mseRF.append(media_mseRF)
rmseRF.append(media_rmseRF)

dfRF = pd.DataFrame({
    'criterion':p_RFcriterion,
    'n_estimators':p_RFn_estimators,
    'min_samples_leaf':p_RFmin_samples_leaf,
    'MAE': maeRF,
    'MSE': mseRF,
    'RMSE': rmseRF
})

print("O ÚLTIMO VALOR (20) É A MÉDIA !!!")

display(dfRF)
dfRF.to_csv('resultadosModelos/dfRF.csv', sep = ';', index = False, encoding='UTF-8')
