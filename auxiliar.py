#BG
from BGboost import BGBRegressor


mseBG = []
rmseBG= []
maeBG= []

p_BGbooster = []
p_BGestimators = []


param_grid = {'booster': ['gblinear', 'gbtree'], 'n_estimators':[100,200,500,1000]}

melhor_validacao_BG = 1000
best_paramsBG = {}



for i in range(20):
    print(i+1)
    for booster in param_grid['booster']:
        for estimators in param_grid['n_estimators']:
            BG = BGBRegressor(booster=booster, n_estimators=estimators)
            BG.fit(x_treino, y_treino)
            opiniao = BG.predict(x_teste)
            mse_validacao_BG = mean_squared_error(y_teste, opiniao)
            
            if (mse_validacao_BG < melhor_validacao_BG):
                melhor_validacao_BG = mse_validacao_BG
                best_paramsBG = {
                    'booster': booster,
                    'n_estimators':estimators
                }


    print(melhor_validacao_BG)
    print(mse_validacao_BG)
    print(best_paramsBG)


    BG = BGBRegressor(booster=best_paramsBG['booster'], n_estimators=best_paramsBG['n_estimators'])
    
    BG.fit(x_treino, y_treino)
    opiniao_BG = BG.predict(x_teste)

    mae = mean_absolute_error(y_teste, opiniao_BG).round(2)
    mse = mean_squared_error(y_teste, opiniao_BG).round(2)
    rmse = np.sqrt(mse).round(2)

    maeBG.append(mae)
    mseBG.append(mse)
    rmseBG.append(rmse)

    p_BGbooster.append(best_paramsBG['booster'])
    p_BGestimators.append(best_paramsBG['n_estimators'])

# RESULTADOS !!!!
maxp_BGbooster = max(set(p_BGbooster), key=p_BGbooster.count)
maxp_BGestimators = max(set(p_BGestimators), key=p_BGestimators.count)



media_maeBG = np.mean(maeBG)
media_mseBG = np.mean(mseBG)
media_rmseBG = np.mean(rmseBG)

p_BGbooster.append(maxp_BGbooster)
p_BGestimators.append(maxp_BGestimators)


maeBG.append(media_maeBG)
mseBG.append(media_mseBG)
rmseBG.append(media_rmseBG)

dfBG = pd.DataFrame({
    'booster':p_BGbooster,
    'estimators':p_BGestimators,
    'MAE': maeBG,
    'MSE': mseBG,
    'RMSE': rmseBG
})

print("O ÚLTIMO VALOR (20) É A MÉDIA !!!")

display(dfBG)
dfBG.to_csv('resultadosModelos/dfBG.csv', sep = ';', index = False, encoding='UTF-8')
