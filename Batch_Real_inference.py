import joblib
import pandas as pd
import Piplan_Classes
import __main__

__main__.ColumnDropper = Piplan_Classes.ColumnDropper
__main__.Parse_data = Piplan_Classes.Parse_data
__main__.Melt_data = Piplan_Classes.Melt_data
__main__.Drop_na = Piplan_Classes.Drop_na
__main__.Extract_machine_id = Piplan_Classes.Extract_machine_id
__main__.Sort_For_Machine = Piplan_Classes.Sort_For_Machine
__main__.Calculate_power_diff = Piplan_Classes.Calculate_power_diff
__main__.Sort = Piplan_Classes.Sort

columns_to_drop = ["cet_cest_timestamp","area_offices","area_room_1",
                   "area_room_2","area_room_3","area_room_4","compressor",
                   "cooling_aggregate","cooling_pumps","dishwasher","ev",
                   "grid_import","pv_facade","pv_roof","refrigerator","ventilation"]
normal_column = ["utc_timestamp", "machine_col", "power", "machine_id", "power_diff"]
training_data = ["utc_timestamp","power_diff"]



model = joblib.load("Isolation_Forest_Model3.joblib")
Final_piplan = joblib.load("preprocessing_pipeline_new2.joblib")


def predict_batch(test_batch):

    if not isinstance(test_batch, pd.DataFrame):
        if hasattr(test_batch, 'dict'): test_batch = pd.DataFrame(test_batch)
        elif isinstance(test_batch, dict): test_batch = pd.DataFrame([test_batch])
        elif isinstance(test_batch, list): test_batch = pd.DataFrame(test_batch)
        else: raise ValueError("Input format not supported. Expected: DataFrame, dict, or list of dicts.")

    transformed_data = Final_piplan.fit_transform(test_batch)

    col_names =  training_data + [ 'machine_1', 'machine_2', 'machine_3', 'machine_4', 'machine_5']

    transformed_df = pd.DataFrame(transformed_data, columns=col_names)

    transformed_df_for_prediction = transformed_df.drop(columns=['utc_timestamp'])

    predictions = model.predict(transformed_df_for_prediction)
    predictions = (predictions == -1).astype(int)

    final_df = pd.DataFrame()
    if 'utc_timestamp' in transformed_df.columns:
        # Si c'est déjà un float/int, on convertit en datetime lisible
        final_df['utc_timestamp'] = pd.to_datetime(transformed_df['utc_timestamp'], unit='s', errors='coerce')
    else:
        final_df['utc_timestamp'] = pd.Timestamp.now()  # Valeur par défaut si perdu

    final_df['machine_id'] = transformed_df[
        [col for col in transformed_df.columns if col.startswith('machine_')]].idxmax(axis=1).str.replace(
        'machine_', '').astype(int)

    final_df['anomaly'] = predictions
    if 'anomaly' in final_df.columns:
        final_df['anomaly'] = final_df['anomaly'].astype(int)

    final_df['utc_timestamp'] = pd.to_datetime(transformed_df['utc_timestamp'], unit='s')

    final_df['machine_id'] = transformed_df[
        [col for col in transformed_df.columns if col.startswith('machine_')]].idxmax(axis=1).str.replace('machine_', '').astype(int)
    final_df['anomaly'] = predictions

    return final_df


def predict_real_time2(new_data):


        if not isinstance(new_data, pd.DataFrame):
            if hasattr(new_data, 'dict'):  # Cas Pydantic
                new_data = pd.DataFrame([new_data.dict()])
            elif isinstance(new_data, dict):  # Cas Dictionnaire
                new_data = pd.DataFrame([new_data])
            elif isinstance(new_data, list):  # Cas Liste de Dictionnaires
                new_data = pd.DataFrame(new_data)
            else:
                raise ValueError("Input format not supported. Expected: DataFrame, dict, or list of dicts.")

        transformed_data = Final_piplan.fit_transform(new_data)

        col_names = training_data + ['machine_1', 'machine_2', 'machine_3', 'machine_4', 'machine_5']

        transformed_df = pd.DataFrame(transformed_data, columns=col_names)

        # drop 'utc_timestamp'
        if 'utc_timestamp' in transformed_df.columns:
            features_for_model = transformed_df.drop(columns=['utc_timestamp'])
        else:

            features_for_model = transformed_df


        prediction = model.predict(features_for_model)
        prediction = (prediction == -1).astype(int)

        final_df = pd.DataFrame()

        if 'utc_timestamp' in transformed_df.columns:
            final_df['utc_timestamp'] = pd.to_datetime(transformed_df['utc_timestamp'], unit='s', errors='coerce')
        else:
            final_df['utc_timestamp'] = pd.Timestamp.now()


        final_df['machine_id'] = transformed_df[
            [col for col in transformed_df.columns if col.startswith('machine_')]].idxmax(axis=1).str.replace(
            'machine_', '').astype(int)

        final_df['anomaly'] =  prediction
        if 'anomaly' in final_df.columns:
            final_df['anomaly'] = final_df['anomaly'].astype(int)



        return final_df

