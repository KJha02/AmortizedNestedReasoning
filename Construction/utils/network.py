import torch


def update_network(loss, optimizer, scaler=None, model = None):
    """update network parameters"""
    if model is not None:
        for param in model.parameters():
            param.grad = None
    else:
        optimizer.zero_grad()
    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward(retain_graph=True)
        optimizer.step()


def save_model(model, path):
    """save trained model parameters"""
    torch.save(model.state_dict(), path)


def load_model(model, path, device=None):
    """load trained model parameters"""
    if device is not None:
        model.load_state_dict(dict(torch.load(path, map_location=device)))
    else:
        model.load_state_dict(dict(torch.load(path)))


def expand_vec(vec_tensor, output_dim):
    """expand vec tensor spatially"""
    return (
        vec_tensor.repeat(1, output_dim[1] * output_dim[2])
        .view(output_dim[1], output_dim[2], output_dim[0])
        .permute(2, 0, 1)
        .unsqueeze(0)
    )


def expand_batch(batch_tensor, output_dim):
    """expand batch spatially"""
    batch_size = batch_tensor.shape[0]
    return (
        batch_tensor.repeat(1, output_dim[0] * output_dim[1])
        .view(batch_size, output_dim[0], output_dim[1], output_dim[2])
        .permute(0, 3, 1, 2)
        # .unsqueeze(0)
    )
