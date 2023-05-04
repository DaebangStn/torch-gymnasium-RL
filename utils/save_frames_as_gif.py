from matplotlib import animation
import matplotlib.pyplot as plt
import pickle
import os.path
import logging.config

frame_pickle_path = 'C:/Users/geon/PycharmProjects/torch-gymnasium-RL/logs/2023-05-02-11-06-39-CartPoleReward-v1.pth' \
                    '.pkl'


def save_frames_as_gif(path, _frames, _logger, save_pickle=True):
    if save_pickle:
        path_frames = os.path.splitext(path)[0] + '.pkl'
        _logger.info(f'save frames as gif to: {path}. frames: {path_frames}')
        with open(path_frames, 'wb') as f:
            pickle.dump(_frames, f)
    else:
        _logger.info(f'save frames as gif: {path}')

    plt.figure(figsize=(_frames[0].shape[1] / 72.0, _frames[0].shape[0] / 72.0), dpi=72)
    patch = plt.imshow(_frames[0])
    plt.axis('off')

    text_annotation = plt.annotate(
        "Frame: 0", (300, 0), fontsize=12, color="black", fontweight="bold"
    )

    def animate(i):
        if i % 100 == 0:
            _logger.info(f"save frames {i}/{len(_frames)}")
        # overlay text to the current patch
        patch.set_data(_frames[i])
        text_annotation.set_text(f"Frame: {i}")

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(_frames), interval=50)
    _logger.info(f'saving as gif started.')
    anim.save(path, writer='imagemagick', fps=60)


if __name__ == '__main__':
    logging.basicConfig(level=logging.ERROR)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not os.path.exists(frame_pickle_path):
        logger.error(f'frame pickle file not found: {frame_pickle_path}. run test.py first')
        exit(1)

    with open(frame_pickle_path, 'rb') as f:
        frames = pickle.load(f)

    logger.info(f'loaded frame numbers: {len(frames)}')

    save_frames_as_gif(os.path.splitext(frame_pickle_path)[0] + '.gif', frames, logger, False)
