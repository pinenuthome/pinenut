"""Train and visualize an MLP on the spiral classification dataset."""

import argparse
import json
from pathlib import Path

import numpy as np
from matplotlib import animation, pyplot as plt

import pinenut as pn
from pinenut import nn, optim
from pinenut.datasets import SpiralDataset
from pinenut.nn import functional as F
from pinenut.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--samples-per-class', type=int, default=100)
    parser.add_argument('--classes', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=30)
    parser.add_argument('--hidden-size', type=int, default=50)
    parser.add_argument('--learning-rate', type=float, default=0.01)
    parser.add_argument('--device', choices=('auto', 'cpu', 'gpu'), default='auto')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--grid-step', type=float, default=0.025)
    parser.add_argument('--log-every', type=int, default=10)
    parser.add_argument('--output-dir', default='demo_outputs/spiral')
    parser.add_argument('--animate', action='store_true')
    parser.add_argument('--snapshot-every', type=int, default=10)
    parser.add_argument('--fps', type=int, default=2)
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--no-save-state', action='store_true')
    args = parser.parse_args()

    positive_values = {
        '--epochs': args.epochs,
        '--samples-per-class': args.samples_per_class,
        '--batch-size': args.batch_size,
        '--hidden-size': args.hidden_size,
        '--log-every': args.log_every,
        '--snapshot-every': args.snapshot_every,
        '--fps': args.fps,
    }
    for name, value in positive_values.items():
        if value <= 0:
            parser.error('{} must be greater than zero'.format(name))
    if args.classes < 2:
        parser.error('--classes must be at least two')
    if args.learning_rate <= 0:
        parser.error('--learning-rate must be greater than zero')
    if args.grid_step <= 0:
        parser.error('--grid-step must be greater than zero')
    return args


def visible_gpu_count():
    return pn.cuda.device_count()


def select_device(requested):
    gpu_count = visible_gpu_count()
    if requested == 'gpu' and gpu_count == 0:
        raise RuntimeError('GPU requested, but no CUDA-capable device is available')
    return requested == 'gpu' or (requested == 'auto' and gpu_count > 0)


def as_float(value):
    return float(np.asarray(pn.cuda.as_numpy(value)))


def train_epoch(model, dataloader, optimizer):
    total_loss = 0.0
    total_accuracy = 0.0
    sample_count = 0

    for data, labels in dataloader:
        optimizer.zero_grad()
        logits = model(data)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()

        batch_count = len(labels)
        batch_accuracy = F.accuracy(logits, labels)
        total_loss += as_float(loss.data) * batch_count
        total_accuracy += as_float(batch_accuracy.data) * batch_count
        sample_count += batch_count

        loss.unchain_backward()

    return total_loss / sample_count, total_accuracy / sample_count


def create_prediction_grid(data, step):
    margin = 0.1
    x_min, x_max = data[:, 0].min() - margin, data[:, 0].max() + margin
    y_min, y_max = data[:, 1].min() - margin, data[:, 1].max() + margin
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, step),
        np.arange(y_min, y_max, step),
    )
    points = np.column_stack((xx.ravel(), yy.ravel()))
    return xx, yy, points


def predict_classes(model, points, use_gpu):
    model_input = pn.cuda.as_cupy(points) if use_gpu else points
    with pn.no_grad():
        logits = model(model_input)
    predictions = pn.cuda.as_numpy(logits.data).argmax(axis=1)
    return predictions


def draw_demo(axes, dataset, grid, predictions, history, epoch, classes):
    decision_ax, loss_ax, accuracy_ax = axes
    for axis in axes:
        axis.clear()

    xx, yy = grid
    class_map = predictions.reshape(xx.shape)
    color_map = plt.get_cmap('Spectral', classes)
    levels = np.arange(classes + 1) - 0.5

    decision_ax.contourf(
        xx, yy, class_map, levels=levels, cmap=color_map, alpha=0.45)
    decision_ax.scatter(
        dataset.data[:, 0], dataset.data[:, 1], c=dataset.label,
        cmap=color_map, vmin=-0.5, vmax=classes - 0.5,
        edgecolors='white', linewidths=0.45, s=22)
    decision_ax.set_title('Decision boundary · epoch {}'.format(epoch))
    decision_ax.set_xlabel('x₁')
    decision_ax.set_ylabel('x₂')

    visible_epochs = [value for value in history['epoch'] if value <= epoch]
    visible_count = len(visible_epochs)
    loss_ax.plot(
        visible_epochs, history['loss'][:visible_count], color='#d95f02')
    loss_ax.set_title('Training loss')
    loss_ax.set_xlabel('Epoch')
    loss_ax.set_ylabel('Cross entropy')
    loss_ax.grid(alpha=0.25)

    accuracy_ax.plot(
        visible_epochs, history['accuracy'][:visible_count], color='#1b9e77')
    accuracy_ax.set_title('Training accuracy')
    accuracy_ax.set_xlabel('Epoch')
    accuracy_ax.set_ylabel('Accuracy')
    accuracy_ax.set_ylim(0.0, 1.0)
    accuracy_ax.grid(alpha=0.25)


def save_final_figure(path, dataset, grid, predictions, history, classes, show):
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    draw_demo(
        axes, dataset, grid, predictions, history,
        history['epoch'][-1], classes)
    figure.suptitle('Pinenut spiral classification', fontsize=14)
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(figure)


def save_animation(path, dataset, grid, snapshots, history, classes, fps):
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    figure.suptitle('Pinenut spiral classification', fontsize=14)

    def update(frame_index):
        snapshot = snapshots[frame_index]
        draw_demo(
            axes, dataset, grid, snapshot['predictions'], history,
            snapshot['epoch'], classes)
        figure.tight_layout()

    demo_animation = animation.FuncAnimation(
        figure, update, frames=len(snapshots), interval=1000 // fps,
        repeat_delay=1200, blit=False)
    demo_animation.save(
        path, writer=animation.PillowWriter(fps=fps), dpi=110)
    plt.close(figure)


def save_metrics(path, args, device, history):
    payload = {
        'device': device,
        'config': vars(args),
        'history': history,
    }
    with path.open('w', encoding='utf-8') as file:
        json.dump(payload, file, indent=2)


def main():
    args = parse_args()
    use_gpu = select_device(args.device)
    device = 'gpu' if use_gpu else 'cpu'

    np.random.seed(args.seed)
    if use_gpu:
        pn.cuda.cupy().random.seed(args.seed)

    dataset = SpiralDataset(
        args.samples_per_class,
        args.classes,
        xdatatype=np.float32,
        ydatatype=np.int32,
    )
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True)
    model = nn.MLP(
        2,
        (args.hidden_size, args.hidden_size, args.classes),
        hidden_activation=F.relu,
    )
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    if use_gpu:
        model.cuda()
        dataloader.cuda()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    grid_x, grid_y, grid_points = create_prediction_grid(
        dataset.data, args.grid_step)
    history = {'epoch': [], 'loss': [], 'accuracy': []}
    snapshots = []

    def capture_snapshot(epoch):
        predictions = predict_classes(model, grid_points, use_gpu)
        snapshots.append({'epoch': epoch, 'predictions': predictions})

    if args.animate:
        capture_snapshot(0)

    print('Training on {} with {} samples'.format(device, len(dataset)))
    for epoch in range(1, args.epochs + 1):
        loss, epoch_accuracy = train_epoch(model, dataloader, optimizer)
        history['epoch'].append(epoch)
        history['loss'].append(loss)
        history['accuracy'].append(epoch_accuracy)

        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(
                'epoch {:>3}/{} · loss {:.4f} · accuracy {:.2%}'.format(
                    epoch, args.epochs, loss, epoch_accuracy))

        should_capture = (
            args.animate
            and (epoch % args.snapshot_every == 0 or epoch == args.epochs)
        )
        if should_capture:
            capture_snapshot(epoch)

    final_predictions = predict_classes(model, grid_points, use_gpu)
    figure_path = output_dir / 'spiral_training.png'
    metrics_path = output_dir / 'spiral_metrics.json'
    state_path = output_dir / 'spiral_state.npz'
    animation_path = output_dir / 'spiral_training.gif'

    save_final_figure(
        figure_path, dataset, (grid_x, grid_y), final_predictions,
        history, args.classes, args.show)
    save_metrics(metrics_path, args, device, history)
    if not args.no_save_state:
        pn.save(model.state_dict(), state_path)
    if args.animate:
        save_animation(
            animation_path, dataset, (grid_x, grid_y), snapshots,
            history, args.classes, args.fps)

    print('Saved figure:  {}'.format(figure_path))
    print('Saved metrics: {}'.format(metrics_path))
    if not args.no_save_state:
        print('Saved state:   {}'.format(state_path))
    if args.animate:
        print('Saved animation: {}'.format(animation_path))


if __name__ == '__main__':
    main()
