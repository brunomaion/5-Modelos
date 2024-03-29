janela = range(1, 20)

train = df[:split]
test = df[split:]

mseARIMA = []
rmseARIMA = []
maeARIMA = []
parametrosARIMA = []
auxMae = 0


for j in range(20):

    maeCalibra = 0
    auxMae = 0
    ## CALIBRAR
    for i in janela:
        media_movel = df['CasosDia'].rolling(i).mean()
        media_movel[split:] = media_movel[split-1]

        maeCalibra = mean_absolute_error(test['CasosDia'], media_movel[split:]).round(2)

        if (maeCalibra < auxMae or auxMae == 0):
            auxMae = maeCalibra
            para_ARIMA = i



    ## TREINA
    media_movel = df['CasosDia'].rolling(para_ARIMA).mean()
    media_movel[split:] = media_movel[split-1]

    mae = mean_absolute_error(test['CasosDia'], media_movel[split:]).round(2)
    mse = mean_squared_error(test['CasosDia'], media_movel[split:]).round(2)
    rmse = np.sqrt(mse).round(2)

    mseARIMA.append(mse)
    rmseARIMA.append(rmse)
    maeARIMA.append(mae)
    parametrosARIMA.append(para_ARIMA)


# RESULTADOSSSS !!!!
    
parametroMaisEscolhidoARIMA = max(set(parametrosARIMA), key=parametrosARIMA.count)
media_maeARIMA = np.mean(maeARIMA)
media_mseARIMA = np.mean(mseARIMA)
media_rmseARIMA = np.mean(rmseARIMA)

parametrosARIMA.append(parametroMaisEscolhidoARIMA)
maeARIMA.append(media_maeARIMA)
mseARIMA.append(media_mseARIMA)
rmseARIMA.append(media_rmseARIMA)


dfMediaMovel = pd.DataFrame({
    'PARA': parametrosARIMA,
    'MAE': maeARIMA,
    'MSE': mseARIMA,
    'RMSE': rmseARIMA
})


print("!! ULTIMO VALOR (20) É A MÉDIA !!!")
display(dfMediaMovel)



plt.figure(figsize=(20,5))
plt.grid()
plt.plot(train['CasosDia'], label='Train')
plt.plot(test['CasosDia'], label='Test')
plt.plot(media_movel, label='Simple moving average forecast')
plt.legend(loc='best')
plt.title('Simple Moving Average Method')
plt.show()

plt.figure(figsize=(20,5))
plt.grid()
plt.plot(test['CasosDia'], label='Test')
plt.plot(media_movel[split:], label='Simple moving average forecast')
plt.legend(loc='best')
plt.title('Simple Moving Average Method')
plt.show()