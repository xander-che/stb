import os
import time
import pandas as pd
import telegram
import schedule
import asyncio
import logging
from tinkoff.invest import Client, CandleInterval, HistoricCandle
from tinkoff.invest.utils import now
from datetime import timedelta, datetime, timezone
from ta.trend import MACD, EMAIndicator
from ta.momentum import RSIIndicator
from adata import final_df_columns

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

INVEST_TOKEN = os.environ["T_SAND_BOX"]
BOT_TOKEN = os.environ["BOT_TOKEN"]
LOG_BOT_TOKEN = os.environ["LOG_BOT_TOKEN"]
B_ID = os.environ["B_ID"]
CHAT_ID = os.environ["CHAT_ID"]
INTERVALS = [(CandleInterval.CANDLE_INTERVAL_HOUR, '1', 168), (CandleInterval.CANDLE_INTERVAL_4_HOUR, '4', 720)]


def get_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(filename='trade_bot_log.log', mode='w')
    formatter = logging.Formatter('%(name)s %(asctime)s %(levelname)s %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def async_loop(async_function):
    # loop = asyncio.get_event_loop()
    # loop.run_until_complete(async_function)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(async_function)


def cast_money(v):
    return v.units + v.nano / 1e9


def create_df(candles: [HistoricCandle], name: str):
    df = pd.DataFrame([{
        'fname': name,
        'time': c.time,
        'volume': c.volume,
        'open': cast_money(c.open),
        'close': cast_money(c.close),
        'high': cast_money(c.high),
        'low': cast_money(c.low),
        'is_complete': c.is_complete
    } for c in candles])
    return df


def get_name(row: pd.Series):
    return row.fname


def get_ticker(row: pd.Series):
    return row.ticker


def get_type(row: pd.Series):
    return row.type


def get_close(row: pd.Series):
    return row.close


def get_interval(row: pd.Series):
    return row.interval


def check_date(data_today):
    today = datetime.today()
    result = False
    if data_today.date() == today.date():
        result = True
    return result


def check_4h_candle(data_today):
    result = False
    four_h_ago = datetime.now(timezone.utc) - timedelta(hours=4)
    if data_today.hour == four_h_ago.hour:
        result = True
    return result


def get_ema_short(row: pd.Series, epsilon: float):  # epsilon = 0.0001
    res = False
    if abs(row.ema - row.close) > epsilon:
        if row.ema - row.close > 0:
            res = True
    return res


def get_ema_long(row: pd.Series, epsilon: float):  # epsilon = 0.0001
    res = False
    if abs(row.ema - row.close) > epsilon:
        if row.ema - row.close < 0:
            res = True
    return res


def get_macd_short(row: pd.Series, epsilon: float):
    res = False
    if abs(row.macd) - abs(row.macd_signal) < epsilon:
        if (row.previous_macd_signal > 0 and row.macd_signal) > 0 or \
                (row.previous_macd_signal < 0 and row.macd_signal < 0) or \
                (row.previous_macd_signal > 0 > row.macd_signal):
            if row.previous_macd_signal > row.macd_signal:
                res = True
        if row.previous_macd_signal < 0 < row.macd_signal:
            res = True
    return res


def get_macd_long(row: pd.Series, epsilon: float):
    res = False
    if abs(row.macd) - abs(row.macd_signal) < epsilon:
        if (row.previous_macd_signal > 0 and row.macd_signal) > 0 or \
                (row.previous_macd_signal < 0 and row.macd_signal < 0) or \
                (row.previous_macd_signal > 0 > row.macd_signal):
            if row.previous_macd_signal < row.macd_signal:
                res = True
        if row.previous_macd_signal > 0 > row.macd_signal:
            res = True
    return res


def get_rsi_short(row: pd.Series):
    res = False
    if row.previous_rsi > 70 > row.rsi:
        res = True
    return res


def get_rsi_long(row: pd.Series):
    res = False
    if row.previous_rsi < 30 < row.rsi:
        res = True
    return res


def get_signal(final_df: pd.DataFrame):
    res = list()
    for i in range(len(final_df)):
        row = final_df.iloc[i, :]
        tf = get_interval(row)
        close = get_close(row)
        ticker = get_ticker(row)
        asset_type = get_type(row)
        name = get_name(row)
        data_today = final_df.iloc[i]['time']
        close_datetime3 = final_df.iloc[i]['time'] + timedelta(hours=3)
        close_date = str(close_datetime3.date())
        close_time = close_datetime3.time()
        if close_time.minute == 0:
            close_datetime = f'{close_date} {close_time.hour}:00'
        else:
            close_datetime = f'{close_date} {close_time.hour}:{close_time.minute}'
        date_check = check_date(data_today)
        if tf == '4':
            hour_check = check_4h_candle(data_today)
        else:
            hour_check = True
        ema_short = get_ema_short(row, 0.0001)
        ema_long = get_ema_long(row, 0.0001)
        macd_short = get_macd_short(row, 0.05)
        macd_long = get_macd_long(row, 0.05)
        rsi_short = get_rsi_short(row)
        rsi_long = get_rsi_long(row)
        if ema_short is True and macd_short is True and rsi_short is True and date_check is True and hour_check is True:
            res.append([name, tf, 'SHORT', close_datetime, ticker, asset_type, close])
        elif ema_long is True and macd_long is True and rsi_long is True and date_check is True and hour_check is True:
            res.append([name, tf, 'LONG', close_datetime, ticker, asset_type, close])
    return res


def get_futures_candles(futures, client, data: list, logger):
    for interval in INTERVALS:
        counter = 0
        for item in futures.instruments:
            if item.asset_type == 'TYPE_COMMODITY':
                try:
                    candle = client.market_data.get_candles(figi=item.figi,
                                                            from_=now() - timedelta(hours=interval[2]),
                                                            to=now(),
                                                            interval=interval[0])
                except Exception as exc:
                    if counter == 0:
                        logger.exception(f': {exc}')
                    candle = None
                    counter += 1

                if candle is not None:
                    if candle.candles.__sizeof__() > 0:
                        df = create_df(candle.candles, item.name)
                        if len(df) >= 40:
                            if not df.is_complete.iloc[-1]:
                                df = df.drop(axis=0, index=len(df) - 1)
                            if df.is_complete.iloc[-1]:
                                df = df.drop('is_complete', axis=1)
                                df['interval'] = interval[1]
                                # df['ma40'] = EMAIndicator(close=df['close'], window=40).ema_indicator()
                                df['ticker'] = item.ticker
                                df['type'] = 'futures'
                                df['ema'] = EMAIndicator(close=df['close'], window=10).ema_indicator()
                                df['macd'] = MACD(close=df['close']).macd()
                                df['macd_signal'] = MACD(close=df['close']).macd_signal()
                                df['rsi'] = RSIIndicator(close=df['close']).rsi()
                                check_counter = 0
                                tail = df.iloc[-1, :].values.tolist()
                                for value in tail:
                                    if pd.isna(value) is True:
                                        check_counter += 1
                                if check_counter == 0:
                                    previous_macd_signal = df.iloc[-2, -2]
                                    previous_rsi = df.iloc[-2, -1]
                                    tail.insert(-1, previous_macd_signal)
                                    tail.append(previous_rsi)
                                    data.append(tail)
    return data


def get_shares_candles(shares, client, data: list, logger):
    for interval in INTERVALS:
        counter = 0
        for item in shares.instruments:
            if item.currency == 'rub':
                try:
                    candle = client.market_data.get_candles(figi=item.figi,
                                                            from_=now() - timedelta(hours=interval[2]),
                                                            to=now(),
                                                            interval=interval[0])
                except Exception as exc:
                    if counter == 0:
                        logger.exception(f': {exc}')
                    candle = None
                    counter += 1
                if candle is not None:
                    if candle.candles.__sizeof__() > 0:
                        df = create_df(candle.candles, item.name)
                        if len(df) >= 40:
                            if not df.is_complete.iloc[-1]:
                                df = df.drop(axis=0, index=len(df) - 1)
                            if df.is_complete.iloc[-1]:
                                df = df.drop('is_complete', axis=1)
                                df['interval'] = interval[1]
                                # df['ma40'] = EMAIndicator(close=df['close'], window=40).ema_indicator()
                                df['ticker'] = item.ticker
                                df['type'] = 'shares'
                                df['ema'] = EMAIndicator(close=df['close'], window=10).ema_indicator()
                                df['macd'] = MACD(close=df['close']).macd()
                                df['macd_signal'] = MACD(close=df['close']).macd_signal()
                                df['rsi'] = RSIIndicator(close=df['close']).rsi()
                                check_counter = 0
                                tail = df.iloc[-1, :].values.tolist()
                                for value in tail:
                                    if pd.isna(value) is True:
                                        check_counter += 1
                                if check_counter == 0:
                                    previous_macd_signal = df.iloc[-2, -2]
                                    previous_rsi = df.iloc[-2, -1]
                                    tail.insert(-1, previous_macd_signal)
                                    tail.append(previous_rsi)
                                    data.append(tail)
    return data


def get_hour_data(client, logger):
    futures = client.instruments.futures()
    shares = client.instruments.shares()
    data = list()
    data = get_futures_candles(futures, client, data, logger)
    data = get_shares_candles(shares, client, data, logger)
    final_df = pd.DataFrame(data, columns=final_df_columns)
    return final_df


async def send_to_bot(response: list, logger):
    if len(response) == 0:
        response = [["test", "message", "for bot", "00", "00", "00", "00"]]
    bot = telegram.Bot(token=BOT_TOKEN)
    for item in response:
        msg = list()
        header = '--TRIO--\n'
        name = f'<b>{item[0]}</b>\n'
        tf = f'Time Frame = {item[1]} h\n'
        stype = f'Signal type = {item[2]}\n'
        date_time = f'{item[3]}\n'
        ticker = f'Ticker = {item[4]}\n'
        tp = f'Type = {item[5]}\n'
        close = f'Close = {item[6]:.4f}\n'
        msg.append(header)
        msg.append(date_time)
        msg.append(name)
        msg.append(tp)
        msg.append(ticker)
        msg.append(tf)
        msg.append(stype)
        msg.append(close)
        msg.append('\n')

        text = ''.join(msg)
        async with bot:
            try:
                await bot.send_message(chat_id=CHAT_ID, text=text, parse_mode=telegram.constants.ParseMode.HTML)
            except telegram.error.BadRequest as exc:
                logger.exception(f': {exc}')
            time.sleep(3)


async def send_logs(logger):
    logs = list()
    with open("trade_bot_log.log") as log:
        for line in log:
            logs.append(line[:-1])
    if len(logs) == 0:
        logs.append('No logs INFO')
    log_bot = telegram.Bot(token=LOG_BOT_TOKEN)
    msg = list()
    msg.append('LOG INFO\n')
    logs = logs[:7]
    for item in logs:
        line = f'{item}\n'
        msg.append(line)
    text = ''.join(msg)
    async with log_bot:
        try:
            await log_bot.send_message(chat_id=B_ID, text=text, parse_mode=telegram.constants.ParseMode.HTML)
        except telegram.error.BadRequest as exc:
            logger.exception(f': {exc}')


def run():
    logger = get_logger()
    logger.info(': Run job')
    with Client(INVEST_TOKEN) as client:
        final_df = get_hour_data(client, logger)
        response = get_signal(final_df)
    if len(response) > 0:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(send_to_bot(response, logger))
        logger.info(f': Signal in data. {len(final_df)} lines in data')
    else:
        logger.info(f': No signals. {len(final_df)} lines in data')
    async_loop(send_logs(logger))
    logger.handlers = []
    del final_df


if __name__ == "__main__":
    try:
        main_logger = get_logger()
        main_logger.info(': Program started')
        async_loop(send_logs(main_logger))
        main_logger.handlers = []
        schedule.every().hour.at(':01').do(run)

        while True:
            try:
                schedule.run_pending()
                time.sleep(10)
            except Exception as e:
                main_logger = get_logger()
                main_logger.exception(f': {e}')
                async_loop(send_logs(main_logger))
                main_logger.handlers = []

            main_logger.handlers = []
    except Exception as e:
        main_logger = get_logger()
        main_logger.exception(f': {e}')
        async_loop(send_logs(main_logger))
        main_logger.handlers = []
