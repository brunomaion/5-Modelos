#XG


mseXG = []
rmseXG= []
maeXG= []

p_XGbooster = []

param_grid = {'booster': ['gblinear', 'gbtree']}

melhor_validacao_XG = 1000
best_paramsXG = {}



for i in range(20):
    print(i+1)
    for booster in param_grid['booster']:
        XG = XGBRegressor(booster=booster)
        XG.fit(x_treino, y_treino)
        opiniao = XG.predict(x_teste)
        mse_validacao_XG = mean_squared_error(y_teste, opiniao)
        
        if (mse_validacao_XG < melhor_validacao_XG):
            melhor_validacao_XG = mse_validacao_XG
            best_paramsXG = {
                'booster': booster
            }


    print(melhor_validacao_XG)
    print(mse_validacao_XG)
    print(best_paramsXG)

    XG = XGBRegressor(booster=best_paramsXG['booster'])
    
    XG.fit(x_treino, y_treino)
    opiniao_XG = XG.predict(x_teste)

    mae = mean_absolute_error(y_teste, opiniao_XG).round(2)
    mse = mean_squared_error(y_teste, opiniao_XG).round(2)
    rmse = np.sqrt(mse).round(2)

    maeXG.append(mae)
    mseXG.append(mse)
    rmseXG.append(rmse)

    p_XGbooster.append(best_paramsXG['booster'])


# RESULTADOS !!!!
maxp_XGbooster = max(set(p_XGbooster), key=p_XGbooster.count)


media_maeXG = np.mean(maeXG)
media_mseXG = np.mean(mseXG)
media_rmseXG = np.mean(rmseXG)

p_XGbooster.append(maxp_XGbooster)

maeXG.append(media_maeXG)
mseXG.append(media_mseXG)
rmseXG.append(media_rmseXG)

dfXG = pd.DataFrame({
    'booster':p_XGbooster,
    'MAE': maeXG,
    'MSE': mseXG,
    'RMSE': rmseXG
})

print("O ÚLTIMO VALOR (20) É A MÉDIA !!!")

display(dfXG)
dfXG.to_csv('resultadosModelos/dfXG.csv', sep = ';', index = False, encoding='UTF-8')
