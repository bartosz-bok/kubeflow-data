import kfp
from kfp.v2 import dsl
from kfp.v2.dsl import component
from kfp.v2.dsl import (
    Input,
    Output,
    Artifact,
    Dataset,
    Metrics
)

@component(
    packages_to_install=['pandas']
)
def download_dataframe(url: str, output_csv: Output[Dataset]):
  import pandas as pd

  df_data = pd.read_csv(url)

  df_data.to_csv(output_csv.path, index=False)

@component(
    packages_to_install=['pandas',
                         'scikit-learn']
)
def preprocessing(data_path: Input[Dataset], output_preprocessed_data_x_train: Output[Dataset],
                                             output_preprocessed_data_x_val: Output[Dataset],
                                             output_preprocessed_data_y_train: Output[Dataset],
                                             output_preprocessed_data_y_val: Output[Dataset]):
  import pandas as pd
  from sklearn.model_selection import train_test_split

  df_data = pd.read_csv(data_path.path)

  df_data.drop(['Race'], axis=1, inplace=True)
  df_data.replace(to_replace=['No','No, borderline diabetes', 'Yes', 'Yes (during pregnancy)'], value=[0, 0, 1, 1], inplace=True)
  df_data['Sex'] = df_data['Sex'].map({'Female': 1, 'Male': 0})
  df_data['AgeCategory'] = df_data['AgeCategory'].map({'18-24': 1,'25-29': 2,'30-34': 3, '35-39': 4,'40-44': 5,'45-49': 6,'50-54': 7,  '55-59': 8, '60-64': 9, '65-69': 10, '70-74': 11, '75-79': 12, '80 or older': 13})
  df_data['GenHealth'] = df_data['GenHealth'].map({'Poor': 1, 'Fair': 2, 'Good': 3,'Very good': 4,'Excellent': 5})

  X = df_data.drop(['HeartDisease'], axis=1)
  y = df_data['HeartDisease']

  X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
  
  X_train.to_csv(output_preprocessed_data_x_train.path, index=False)
  X_val.to_csv(output_preprocessed_data_x_val.path, index=False)
  y_train.to_csv(output_preprocessed_data_y_train.path, index=False)
  y_val.to_csv(output_preprocessed_data_y_val.path, index=False)

@component(
    packages_to_install=['pandas',
                         'torch',
                         'scikit-learn']
)
def training_1st(num_epochs: int, preprocessed_data_path_x_train: Input[Dataset],
                                  preprocessed_data_path_x_val: Input[Dataset],
                                  preprocessed_data_path_y_train: Input[Dataset],
                                  preprocessed_data_path_y_val: Input[Dataset],
                                  output_accuracy: Output[Metrics]):
  import torch
  import torch.nn as nn
  import torch.optim as optim
  import pandas as pd
  from sklearn.preprocessing import StandardScaler
  from sklearn.metrics import accuracy_score

  X_train = pd.read_csv(preprocessed_data_path_x_train.path)
  X_val = pd.read_csv(preprocessed_data_path_x_val.path)
  y_train = pd.read_csv(preprocessed_data_path_y_train.path)
  y_val = pd.read_csv(preprocessed_data_path_y_val.path)

  scaler = StandardScaler()
  X_train = scaler.fit_transform(X_train)
  X_val = scaler.transform(X_val)

  X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
  y_train_tensor = torch.tensor(y_train.values.reshape(-1, 1), dtype=torch.float32)
  X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
  y_val_tensor = torch.tensor(y_val.values.reshape(-1, 1), dtype=torch.float32)

  # Definicja modelu
  class RegressionModel(nn.Module):
      def __init__(self):
          super(RegressionModel, self).__init__()
          self.fc1 = nn.Linear(16, 8)
          self.fc2 = nn.Linear(8, 4)
          self.fc3 = nn.Linear(4, 1)

      def forward(self, x):
          x = torch.relu(self.fc1(x))
          x = torch.relu(self.fc2(x))
          x = self.fc3(x)
          return x

  model = RegressionModel()

  criterion = nn.MSELoss()
  optimizer = optim.Adam(model.parameters(), lr=0.01)

  # Trenowanie modelu
  for epoch in range(num_epochs):
      # Forward pass
      outputs = model(X_train_tensor)
      loss = criterion(outputs, y_train_tensor)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      with torch.no_grad():
          val_outputs = model(X_val_tensor)
          val_loss = criterion(val_outputs, y_val_tensor)
          print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Validation Loss: {val_loss.item():.4f}')

  with torch.no_grad():
      val_predictions = model(X_val_tensor)
      val_predictions = torch.round(val_predictions)
      val_accuracy = accuracy_score(y_val_tensor.numpy(), val_predictions.numpy())
      print(f'Validation Accuracy: {val_accuracy:.4f}')

  output_accuracy.log_metric('accuracy', val_accuracy) 

@component(
    packages_to_install=['pandas',
                         'torch',
                         'scikit-learn']
)
def training_2nd(num_epochs: int, preprocessed_data_path_x_train: Input[Dataset],
                                  preprocessed_data_path_x_val: Input[Dataset],
                                  preprocessed_data_path_y_train: Input[Dataset],
                                  preprocessed_data_path_y_val: Input[Dataset],
                                  output_accuracy: Output[Metrics]):
  import torch
  import torch.nn as nn
  import torch.optim as optim
  import pandas as pd
  from sklearn.preprocessing import StandardScaler
  from sklearn.metrics import accuracy_score

  X_train = pd.read_csv(preprocessed_data_path_x_train.path)
  X_val = pd.read_csv(preprocessed_data_path_x_val.path)
  y_train = pd.read_csv(preprocessed_data_path_y_train.path)
  y_val = pd.read_csv(preprocessed_data_path_y_val.path)

  scaler = StandardScaler()
  X_train = scaler.fit_transform(X_train)
  X_val = scaler.transform(X_val)

  X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
  y_train_tensor = torch.tensor(y_train.values.reshape(-1, 1), dtype=torch.float32)
  X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
  y_val_tensor = torch.tensor(y_val.values.reshape(-1, 1), dtype=torch.float32)

  # Definicja modelu
  class RegressionModel(nn.Module):
      def __init__(self):
          super(RegressionModel, self).__init__()
          self.fc1 = nn.Linear(16, 16)
          self.fc2 = nn.Linear(16, 8)
          self.fc3 = nn.Linear(8, 8)
          self.fc4 = nn.Linear(8, 1)

      def forward(self, x):
          x = torch.relu(self.fc1(x))
          x = torch.relu(self.fc2(x))
          x = torch.relu(self.fc3(x))
          x = self.fc4(x)
          return x

  model = RegressionModel()

  criterion = nn.MSELoss()
  optimizer = optim.Adam(model.parameters(), lr=0.01)

  # Trenowanie modelu
  for epoch in range(num_epochs):
      # Forward pass
      outputs = model(X_train_tensor)
      loss = criterion(outputs, y_train_tensor)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      with torch.no_grad():
          val_outputs = model(X_val_tensor)
          val_loss = criterion(val_outputs, y_val_tensor)
          print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Validation Loss: {val_loss.item():.4f}')

  with torch.no_grad():
      val_predictions = model(X_val_tensor)
      val_predictions = torch.round(val_predictions)
      val_accuracy = accuracy_score(y_val_tensor.numpy(), val_predictions.numpy())
      print(f'Validation Accuracy: {val_accuracy:.4f}')

  output_accuracy.log_metric('accuracy', val_accuracy) 

@component()
def compare_accuracy(accuracy_1st: Input[Metrics],accuracy_2nd: Input[Metrics], better_model: Output[Metrics]):

  accuracy_1st_result = accuracy_1st.metadata['accuracy']
  accuracy_2nd_result = accuracy_2nd.metadata['accuracy']

  print(f'Pierwszy model osiagnal wartosc {accuracy_1st_result}, a drugi {accuracy_2nd_result}.')

  which_better = 0

  if accuracy_1st_result > accuracy_2nd_result:
    print('Pierwszy model osiagnal wieksza dokladnosc')
    which_better = 1
  elif accuracy_1st_result < accuracy_2nd_result:
    print('Drugi model osiagnal wieksza dokladnosc')
    which_better = 2
  elif accuracy_1st_result == accuracy_2nd_result:
    print('Obydwa modele osiagnely taka sama dokladnosc')
  else:
    print('Blad')

  better_model.log_metric('better_model', which_better)

# Define a pipeline and create a task from a component:
@dsl.pipeline(
    name='my-pipeline',
    # You can optionally specify your own pipeline_root
    # pipeline_root='gs://my-pipeline-root/example-pipeline',
)
def my_pipeline(url: str, num_epochs_1st: int, num_epochs_2nd: int):
  web_downloader_task = download_dataframe(url=url)
  preprocessing_task = preprocessing(data_path=web_downloader_task.outputs['output_csv'])
  training_task_1st = training_1st(num_epochs=num_epochs_1st, preprocessed_data_path_x_train=preprocessing_task.outputs['output_preprocessed_data_x_train'],
                                                              preprocessed_data_path_x_val=preprocessing_task.outputs['output_preprocessed_data_x_val'],
                                                              preprocessed_data_path_y_train=preprocessing_task.outputs['output_preprocessed_data_y_train'],
                                                              preprocessed_data_path_y_val=preprocessing_task.outputs['output_preprocessed_data_y_val'])
  training_task_2nd = training_2nd(num_epochs=num_epochs_2nd, preprocessed_data_path_x_train=preprocessing_task.outputs['output_preprocessed_data_x_train'],
                                                              preprocessed_data_path_x_val=preprocessing_task.outputs['output_preprocessed_data_x_val'],
                                                              preprocessed_data_path_y_train=preprocessing_task.outputs['output_preprocessed_data_y_train'],
                                                              preprocessed_data_path_y_val=preprocessing_task.outputs['output_preprocessed_data_y_val'])
  compare_task = compare_accuracy(accuracy_1st=training_task_1st.outputs['output_accuracy'], accuracy_2nd=training_task_2nd.outputs['output_accuracy'])

kfp.compiler.Compiler(mode=kfp.dsl.PipelineExecutionMode.V2_COMPATIBLE).compile(
    pipeline_func=my_pipeline,
    package_path='pipeline_distributed_v1.yaml')