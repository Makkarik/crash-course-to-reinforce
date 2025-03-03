"""Misc utilities."""

import os
import pandas as pd
import imageio


def mp4_to_gif(folder: str) -> None:
    """Convert MP4 video to GIF.

    Parameters
    ----------
    folder : str
        The folder containing MP4 files to be converted.

    """
    paths = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".mp4")]
    gif_paths = [p[: p.rfind(".")] + ".gif" for p in paths]

    for video_path, gif_path in zip(paths, gif_paths):
        with imageio.get_reader(video_path) as reader:
            fps = reader.get_meta_data()["fps"]

            writer = imageio.get_writer(gif_path, fps=fps, loop=0)
            for frame in reader:
                writer.append_data(frame)
            writer.close()

        os.remove(video_path)


def postprocess_results(df: pd.DataFrame, results: dict, step_label: str, agent_label: str) -> pd.DataFrame:
    """
    Takes an existing DataFrame (with a 3-level MultiIndex header) and a results
    dictionary. This function creates two new columns—one for reward and one for normalized length—
    by writing each result value directly (no averaging). The header for these new columns is built as:
      Level 0: agent_label (e.g. "Value Iteration")
      Level 1: step_label (e.g. "Training" or "Validation")
      Level 2: metric name ("reward" or "norm_length")

    Since the results contain a number of values (for example 10 values) which might be more
    (or less) than the number of rows in the existing df, we reindex df accordingly so that the new
    columns’ data (one value per row) can be appended “after” the existing rows.

    Parameters:
      df (pd.DataFrame): The original DataFrame.
      results (dict): Dictionary with keys 'reward' and 'norm_length'. The values for 'reward'
                      are assumed to be in a deque (or list) and 'norm_length' is a list.
      step_label (str): Step label such as "Training" or "Validation".
      agent_label (str): Agent label such as "Value Iteration" or "Random Agent".

    Returns:
      pd.DataFrame: The modified DataFrame with two new columns added.
    """
    # Determine number of entries in results (assumed the same for 'reward' and 'norm_length')
    n_results = len(results['reward'])

    # Create a new DataFrame from the results dictionary.
    # We convert the rewards deque to a list to ensure proper handling.
    new_data = {
        (agent_label, step_label, 'reward'): list(results['reward']),
        (agent_label, step_label, 'norm_length'): results['norm_length']
    }
    new_df = pd.DataFrame(new_data, index=range(n_results))

    # Reindex the original df if necessary so that it has at least n_results rows.
    # If df has fewer rows, extra rows will be filled with NaN.
    if len(df) < n_results:
        df = df.reindex(range(n_results))
    # If df has more rows than new_df, reindex new_df to align with df's index.
    elif len(df) > n_results:
        new_df = new_df.reindex(df.index)

    # Concatenate the original DataFrame with the new DataFrame of results along columns.
    df_modified = pd.concat([df, new_df], axis=1)

    return df_modified