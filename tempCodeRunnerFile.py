if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    torch.save(best_model.state_dict(), "./model/best_model.pth")