
def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag
def sample_data(loader):
    while True:
        for batch in loader:
            yield batch
