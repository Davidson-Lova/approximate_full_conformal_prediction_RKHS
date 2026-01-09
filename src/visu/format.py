import seaborn as sns


def set_style():
    # This sets reasonable defaults for font size for
    # a figure that will go in a paper
    sns.set_context("paper")
    # Set the font to be serif, rather than sans
    sns.set(font="serif", font_scale=0.75)
    sns.set_palette("muted")
    # Make the background white, and specify the
    # specific font family
    sns.set_style(
        "whitegrid",
        {"font.family": "serif", "font.serif": ["Times", "Palatino", "serif"]},
    )
