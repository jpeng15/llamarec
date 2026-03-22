import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split
from preprocess_data import MovieLensLoader
from data_loader import MovieLensDataset
from model import MatrixFactorization

# Hyperparameters
N_FACTORS = 64
BATCH_SIZE = 1024
LR = 0.001
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading Data...")
# Initialize Data
load_data = MovieLensLoader("ml-32m/ratings.csv")
df = load_data.df

# Split Data
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

train_dataset = MovieLensDataset(train_df.user_idx.values, train_df.movie_idx.values, train_df.rating.values)
test_dataset = MovieLensDataset(test_df.user_idx.values, test_df.movie_idx.values, test_df.rating.values)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
print(f"Data Loaded. Train: {len(train_df)}, Test: {len(test_df)}")

print("Initializing Model...")
# Initialize Model
model = MatrixFactorization(len(load_data.user_map), len(load_data.movie_map), N_FACTORS).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5) # L2 for regularization
criterion = nn.MSELoss()
print("Model Initialized.")

def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for users, movies, ratings in loader:
            users, movies, ratings = users.to(device), movies.to(device), ratings.to(device)
            output = model(users, movies)
            loss = criterion(output, ratings)
            total_loss += loss.item()
    return (total_loss / len(loader))**0.5

# Training Loop
for epoch in range(EPOCHS):
    model.train()
    print(f"Epoch {epoch+1}/{EPOCHS}")
    total_loss = 0
    for users, movies, ratings in train_loader:
        users, movies, ratings = users.to(DEVICE), movies.to(DEVICE), ratings.to(DEVICE)
        
        optimizer.zero_grad()
        output = model(users, movies)
        loss = criterion(output, ratings)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    train_rmse = avg_loss**0.5
    test_rmse = evaluate(model, test_loader, DEVICE, criterion)
    print(f"Epoch {epoch+1}/{EPOCHS} - Train RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse:.4f}")

print("\nFinal Evaluation...")
final_test_rmse = evaluate(model, test_loader, DEVICE, criterion)
print(f"Final Test RMSE: {final_test_rmse:.4f}")

# Save Model
print("Saving Model...")
torch.save(model.state_dict(), "model.pth")
print("Model saved to model.pth")