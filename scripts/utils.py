import torch
import matplotlib.pyplot as plt
import pandas as pd





def plot_data(dataloader):
    torch.manual_seed(42)
    class_names = dataloader.dataset.classes

    fig = plt.figure(figsize=(9, 9))
    rows, cols = 4, 4
    for i in range(1, rows * cols + 1):
        random_idx = torch.randint(0, len(dataloader.dataset), size=[1]).item()
        img, label = dataloader.dataset[random_idx]
        img = torch.permute(img , (1,2,0))        
        fig.add_subplot(rows, cols, i)
        #, vmax=1, vmin=0
        plt.imshow(img)
        plt.title(class_names[label])
        plt.axis(False)
    plt.savefig('my_dataset.png')
    plt.show()

def get_mean_std(dataLoader):
    mean = 0 
    std = 0
    n_samples = 0

    for images, _ in dataLoader:
        images = images.view(images.shape[0] , images.shape[1] , -1)
        std += images.std(2).sum(0)
        mean += images.mean(2).sum(0)
        n_samples += images.shape[0]

    mean /= n_samples
    std /= n_samples

    return mean,std


def save_model(model, opt , epoch , checkpointPATH):
    checkpoint = {
        'epoch' : epoch,
        'model_state' : model.state_dict(),
        'opt_state' : model.state_dict()
    }
    torch.save(checkpoint , checkpointPATH)
    print("the model saved")


def load_model(model, opt , checkpointPATH):
    checkpoint = torch.load(checkpointPATH)
    model.load_state_dict(checkpoint['model_state'])
    epoch = checkpoint['epoch']
    return epoch

def get_accuracy(loader , model , device):
    model.eval()
    num_correct = 0
    num_samples = 0
   
    with torch.no_grad():
        for x,y in loader:
            x = x.to(device)
            y = y.to(device)

            scores = model(x)
            _, predictions = torch.max(scores , dim=1)
            num_correct += (predictions == y).sum()
            num_samples += x.shape[0]

    model.train()
    return (float(num_correct)/float(num_samples)*100)



def plot_models_together(model_names, train_acc_csv_files, train_loss_csv_files, output_file_path=None):
    num_models = len(model_names)


    """
    model_names = ['Model 1', 'Model 2', 'Model 3', 'Model 4']
    train_acc_csv_files = ['model1_train_acc.csv', 'model2_train_acc.csv', 'model3_train_acc.csv', 'model4_train_acc.csv']
    train_loss_csv_files = ['model1_train_loss.csv', 'model2_train_loss.csv', 'model3_train_loss.csv', 'model4_train_loss.csv']
    output_file_path = 'combined_plot.png'
    
    """



    rows = (num_models + 1) // 2  
    fig, axes = plt.subplots(rows, 2, figsize=(12, 6 * rows))
    fig.suptitle('Training Accuracy and testing Accuracy Over Steps', fontsize=16)

    for i, (model_name, acc_csv, loss_csv) in enumerate(zip(model_names, train_acc_csv_files, train_loss_csv_files)):
        acc_data = pd.read_csv(acc_csv)
        loss_data = pd.read_csv(loss_csv)

        acc_step = acc_data['Step']
        acc_value = acc_data['Value']
        loss_step = loss_data['Step']
        loss_value = loss_data['Value']

        if num_models > 1:
            ax = axes[i // 2, i % 2]
        else:
            ax = axes  # When there's only one model

        ax.plot(acc_step, acc_value, label='Train Accuracy', color='blue')
        ax.plot(loss_step, loss_value, label='test Accuracy', color='red')
        # ax.set_xlabel('Step')
        ax.set_ylabel('Percent')
        ax.set_title(model_name)
        ax.grid(True)

    if num_models > 1:
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if output_file_path:
        plt.savefig(output_file_path, bbox_inches='tight')


    plt.show()







def plot_models_together_in_one(model_names, train_acc_csv_files,  test_acc_csv_files, output_file_path=None):
    num_models = len(model_names)

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle('Training and Testing Accuracy Over Steps', fontsize=16)

    train_colors = ['blue', 'red', 'green', 'orange']
    test_colors = ['lightblue', 'pink', 'lightgreen', 'gold']

    for i, (model_name, train_acc_csv, test_acc_csv) in enumerate(zip(model_names, train_acc_csv_files, test_acc_csv_files)):
        train_acc_data = pd.read_csv(train_acc_csv)
        test_acc_data = pd.read_csv(test_acc_csv)

        train_acc_step = train_acc_data['Step']
        train_acc_value = train_acc_data['Value']
        test_acc_step = test_acc_data['Step']
        test_acc_value = test_acc_data['Value']

        train_color = train_colors[i % len(train_colors)]
        test_color = test_colors[i % len(test_colors)]

        ax.plot(train_acc_step, train_acc_value, label='Train Accuracy - {}'.format(model_name), color=train_color)
        ax.plot(test_acc_step, test_acc_value, label='Test Accuracy - {}'.format(model_name), color=test_color)

    ax.set_xlabel('Step')
    ax.set_ylabel('Percent')
    ax.grid(True)
    
    ax.legend(loc='upper left')
        
    if output_file_path:
        plt.savefig(output_file_path, bbox_inches='tight')

    plt.show()



