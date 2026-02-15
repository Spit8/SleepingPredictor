# --- LIBRAIRIES ---
import pandas as pd
from datetime import timedelta
from ics import Calendar, Event
from prophet import Prophet

# --- CONFIGURATION ---
data_dir = "[directory to data source]"
output_ics = "AwakePeriodsPredictions.ics"

# --- LECTURE DES DONNÉES ---
sleeps = pd.read_csv(data_dir + "UserSleeps_2024-12-29.csv")
scores = pd.read_csv(data_dir + "UserSleepScores_2024-12-29.csv")

# --- NETTOYAGE DES DATES ---
sleeps["start_time"] = pd.to_datetime(sleeps["sleep_start"])
sleeps["end_time"]   = pd.to_datetime(sleeps["sleep_end"])
sleeps = sleeps.sort_values("start_time")

# --- FUSION DES PÉRIODES PROCHES ---
sleeps["gap"] = sleeps["start_time"].diff().dt.total_seconds() / 60
sleeps["merge_group"] = (sleeps["gap"].isna() | (sleeps["gap"] > 30)).cumsum()

sleeps_clean = sleeps.groupby("merge_group").agg(
    start_time=("start_time","min"),
    end_time=("end_time","max")
).reset_index()

# --- DÉTECTION DES PÉRIODES D'ÉVEIL HISTORIQUES ---
awake_periods = pd.DataFrame({
    "start_awake": sleeps_clean["end_time"].shift(),
    "end_awake": sleeps_clean["start_time"]
}).dropna()

awake_periods["duration_min"] = (awake_periods["end_awake"] - awake_periods["start_awake"]).dt.total_seconds()/60
awake_periods = awake_periods[awake_periods["duration_min"] > 30]

# --- PRÉPARATION DES SÉRIES TEMPORELLES ---
scores["score_time"] = pd.to_datetime(scores["score_time"])

# Série pour la durée de sommeil
daily_sleep = scores.groupby(scores["score_time"].dt.date)["sleep_time_minutes"].sum().reset_index()
daily_sleep.columns = ["ds","y"]

# Série pour l’heure de coucher
sleeps["bedtime_hour"] = sleeps["start_time"].dt.hour + sleeps["start_time"].dt.minute/60
daily_bedtime = sleeps.groupby(sleeps["start_time"].dt.date)["bedtime_hour"].mean().reset_index()
daily_bedtime.columns = ["ds","y"]

# --- MODELES PROPHET ---
# Durée de sommeil
model_sleep = Prophet()
model_sleep.fit(daily_sleep)
future_sleep = model_sleep.make_future_dataframe(periods=30)
forecast_sleep = model_sleep.predict(future_sleep)

# Heure de coucher
model_bedtime = Prophet()
model_bedtime.fit(daily_bedtime)
future_bedtime = model_bedtime.make_future_dataframe(periods=30)
forecast_bedtime = model_bedtime.predict(future_bedtime)

# --- CRÉATION D'ÉVÉNEMENTS ICS ---
cal = Calendar()

# Périodes d’éveil historiques
for _, row in awake_periods.iterrows():
    e = Event()
    e.name = "Periode eveil (historique)"
    e.begin = row["start_awake"]
    e.end   = row["end_awake"]
    cal.events.add(e)

# Périodes d’éveil prédictives (basées sur Prophet)
for i in range(30):
    predicted_sleep = forecast_sleep.tail(30).iloc[i]["yhat"]
    predicted_bedtime = forecast_bedtime.tail(30).iloc[i]["yhat"]

    # Construire l'heure de coucher prédite
    sleep_start = pd.to_datetime(forecast_sleep.tail(30).iloc[i]["ds"]) \
                  + timedelta(hours=float(predicted_bedtime))
    sleep_end   = sleep_start + timedelta(minutes=float(predicted_sleep))

    e = Event()
    e.name = "Periode eveil (prediction)"
    e.begin = sleep_end
    e.end   = sleep_start + timedelta(days=1)  # éveil jusqu’au lendemain
    cal.events.add(e)

# --- EXPORT ---
with open(output_ics, "w") as f:
    f.writelines(cal.serialize_iter())

print("✅ Fichier ICS exporté :", output_ics)
