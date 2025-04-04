# Importing dependency packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import talib as ta
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False

# Import Data

# Data includes columns:
# date, open, close, high, low, volume, vix
# Data is S&P 500 index volume price data and implied volatility data
def import_data(path):
    data = pd.read_csv(path)
    data.set_index('date',inplace=True)
    data.index = pd.DatetimeIndex(data.index)
    return data

def plot_close_vix(df):
    plt.figure(figsize=(20,8))
    ax1 = df['close'].plot()
    ax1.legend(['close_price'],loc=1)
    ax2 = ax1.twinx()
    ax2.plot(df['vix'],color='black')
    ax2.legend(['vix'],loc=2)
    plt.grid(True,alpha=0.9)
    plt.show()
    
# Calculation of 20 technical indicators

# SMA_20 - 20-day simple moving average, reflecting medium-term trends\
# EMA_12 - 12-day exponential moving average, responds quickly to price changes
# Bollinger Bands - Upper, middle and lower rails that form a channel for price movement.
# SAR - parabolic turn indicator, dynamically marking stop-loss and take-profit levels
# MACD - Moving Average Convergence Dispersion, to determine the trend strength and reversal
# RSI_14 - 14-day Relative Strength Index, a measure of overbought and oversold conditions.
# Stochastic - SlowK/SlowD to capture short-term price reversal signals
# CCI_20 - 20-day trend indicator, identifies extreme price movements
# Williams - 14-day Williams %R, analyzes overbought and oversold ranges
# ROC_12 - 12-day Rate of Change, measures price acceleration
# ATR_14 - 14-day Average True Volatility, quantifying market volatility
# STD_20 - 20-day standard deviation, reflecting the degree of price dispersion
# DMI indicator - +DI/-DI, determines the contrast between long and short forces
# ADX_14 - 14-day average Convergence Index, measures trend strength
# OBV - Energy Wave, a tool to verify the synchronization of volume and price.
# MFI_14 - 14-day Money Flow Indicator, combining price and volume
# VWAP - Volume Weighted Average Price, benchmark price for intraday trading
# BIAS_20 - 20 Day Deviation Ratio, a measure of price deviation from the SMA
# CMF_20 - 20-day Chia-Ching Money Flow, monitors capital inflows and outflows
# PVT - Volume Trend Indicator, accumulates the effect of volume and price changes
def calculate_technical_indicators(data):
    df = data.copy()
    df['Ln_ret'] = np.log(data['close']/data['close'].shift(1))
    # Trend indicators
    df['SMA_20'] = ta.SMA(df['close'], 20)
    df['EMA_12'] = ta.EMA(df['close'], 12)
    df['Upper_Band'], df['Middle_Band'], df['Lower_Band'] = ta.BBANDS(df['close'], 20, 2, 2)
    df['SAR'] = ta.SAR(df['high'], df['low'], 0.02, 0.2)
    df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = ta.MACD(df['close'], 12, 26, 9)
    
    # Momentum indicators
    df['RSI_14'] = ta.RSI(df['close'], 14)
    df['SlowK'], df['SlowD'] = ta.STOCH(df['high'], df['low'], df['close'], 14, 3, 3)
    df['CCI_20'] = ta.CCI(df['high'], df['low'], df['close'], 20)
    df['WILLR_14'] = ta.WILLR(df['high'], df['low'], df['close'], 14)
    df['ROC_12'] = ta.ROC(df['close'], 12)
    
    # Volatility Indicators
    df['ATR_14'] = ta.ATR(df['high'], df['low'], df['close'], 14)
    df['STD_20'] = ta.STDDEV(df['close'], 20, 1)
    df['PLUS_DI'] = ta.PLUS_DI(df['high'], df['low'], df['close'], 14)
    df['MINUS_DI'] = ta.MINUS_DI(df['high'], df['low'], df['close'], 14)
    df['ADX_14'] = ta.ADX(df['high'], df['low'], df['close'], 14)
    
    # Volume Indicators
    df['OBV'] = ta.OBV(df['close'], df['volume'])
    df['MFI_14'] = ta.MFI(df['high'], df['low'], df['close'], df['volume'], 14)
    df['VWAP'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
    df['BIAS_20'] = (df['close'] - ta.SMA(df['close'], 20)) / ta.SMA(df['close'], 20) * 100
    df['MF_Multiplier'] = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
    df['MF_Volume'] = df['MF_Multiplier'] * df['volume']
    df['CMF_20'] = df['MF_Volume'].rolling(20).sum() / df['volume'].rolling(20).sum()
    df['PriceChange'] = df['close'].pct_change()
    df['PVT'] = (df['PriceChange'] * df['volume']).cumsum()

    # Use the previous day's data to predict the next day's vix
    df['vix'].shift(-1)
    df = df.shift(1).dropna()
    return df

def plot_corr(df):
    plt.figure(figsize=(20,20))
    sns.heatmap(df.corr(),cmap='coolwarm',annot=True,fmt=".2f",linewidths=0.5)

# PCA funcation
def perform_pca(df, variance_threshold=0.95):
    """
    Perform PCA dimensionality reduction
    :param df: DataFrame containing raw data and calculated indicators
    :param tech_columns: List of technical indicator column names for PCA
    :param variance_threshold: Cumulative variance threshold (0-1)
    :return: Principal components, PCA model object
    """    
    # Standardize data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    
    pca = PCA(n_components=variance_threshold).fit(scaled_data)
    pca_data = pca.transform(scaled_data)
    pca_data = pd.DataFrame(data=pca_data,index=df.index,
                            columns=[f"PC{i+1}" for i in range(pca_data.shape[1])])
    return pca,pca_data

def visualize_variance(pca_model):
    """Visualize cumulative explained variance ratio"""
    plt.figure(figsize=(10, 6))
    plt.axhline(y=0.95, color='g', linestyle='--', label='95% Explained Variance')
    plt.plot(range(1,len(pca_model.explained_variance_ratio_)+1),
             np.cumsum(pca_model.explained_variance_ratio_), 'bo-')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('PCA Cumulative Explained Variance Ratio')
    plt.grid(True)
    plt.show()

# Factor Load Thermal Matrix
def plot_pca_loadings(pca_model, df):
    feature_names = df.columns
    """Interpret principal components using factor loadings"""
    loadings = pd.DataFrame(
        pca_model.components_.T,
        columns=[f'PC{i+1}' for i in range(pca_model.n_components_)],
        index=feature_names)
    
    plt.figure(figsize=(20,16))
    sns.heatmap(loadings,cmap='coolwarm', center=0,       
                annot=True,fmt=".2f",linewidths=0.5)
    plt.show()


# LSTM
def buildLSTM(Nh,learnRate=1e-3):
    #输入层
    inputLayer = Input(shape=(1,8))
    #隐藏层
    middle = LSTM(Nh,activation='sigmoid')(inputLayer)
    #输出层 全连接
    outputLayer = Dense(1)(middle)
    #建模
    model = Model(inputs=inputLayer,outputs=outputLayer)
    optimizer = Adam(learning_rate=learnRate)
    model.compile(optimizer=optimizer,loss='mse')
    return model

def loss_result(pred_y,real_y):
    MSE = metrics.mean_squared_error(pred_y, real_y)
    RMSE = metrics.mean_squared_error(pred_y, real_y)**0.5
    MAE = metrics.mean_absolute_error(pred_y, real_y)
    R_2 = metrics.r2_score(pred_y, real_y)
    MAPE = np.mean(np.abs((real_y - pred_y) / real_y)) * 100
    
    result = pd.Series({
        'MSE':MSE,
        'RMSE':RMSE,
        'MAE':MAE,
        'R_2':R_2,
        'MAPE':MAPE
    })
    return result

