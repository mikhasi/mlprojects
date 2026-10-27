import torch
import torch.nn as nn
import torch.optim as optim

from dataset import load_or_download_fashion_mnist, set_seed
from models import FashionMLP, FashionCNN, FashionAttention, count_parameters

# 4

def train_model(model, train_loader, device, epochs=5, lr=1e-3):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"  Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss:.4f}")

def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    accuracy = 100.0 * correct / total
    return accuracy

def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    train_loader, val_loader, test_loader = load_or_download_fashion_mnist(batch_size=128)
    
    models = {
        "MLP": FashionMLP().to(device),
        "CNN": FashionCNN().to(device),
        "Attention": FashionAttention().to(device)
    }
    
    results = {}
    
    print("=" * 60)
    print("Starting Training for 5 Epochs (Adam, lr=1e-3, Batch Size=128)")
    print("=" * 60)
    
    for name, model in models.items():
        print(f"\n--- Training {name} Model ({count_parameters(model):,} parameters) ---")
        train_model(model, train_loader, device, epochs=5, lr=1e-3)
        
        test_acc = evaluate_model(model, test_loader, device)
        results[name] = test_acc
        print(f"-> {name} Final Test Accuracy: {test_acc:.2f}%\n")
        
    print("=" * 60)
    print(f"{'Model':<12} | {'Parameters':<12} | {'Test Accuracy (%)':<18}")
    print("-" * 50)
    for name, acc in results.items():
        params = count_parameters(models[name])
        print(f"{name:<12} | {params:<12,d} | {acc:<18.2f}%")
    print("=" * 60)

if __name__ == "__main__":
    main()