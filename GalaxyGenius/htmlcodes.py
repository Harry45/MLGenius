HTML_TEXT = """
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GalaxyGenius</title>
    <style>
        .container {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            padding: 20px;
        }
        .column {
            flex: 1;
            padding: 10px;
        }
        .column img {
            max-width: 120%;
            height: auto;
        }
        p {
            text-align: justify;
        }
    </style>
</head>
<body>

<div class="container">
    <div class="column">
        <h2>GalaxyGenius</h2>

        <h3>Multi-label Learning for Galaxy Image Tagging</h3>
        <p>Our innovative application utilizes multi-label learning techniques to assign relevant tags to galaxy images. By analyzing various features and characteristics within each image, the system can accurately identify and label multiple attributes such as galaxy type, shape, and size. This approach enhances the efficiency of cataloging vast collections of astronomical data and provides valuable insights into the diverse properties of galaxies across the universe.</p>

        <h3>Multi-Task Learning for Hierarchical Classification of Galaxies</h3>
        <p>Our cutting-edge system employs multi-task learning methodologies to navigate through hierarchical taxonomies and classify galaxies. By simultaneously training on multiple related tasks, the model learns to identify the optimal path down the hierarchical tree structure, enabling precise categorization of galaxies based on their intrinsic properties. This approach streamlines the classification process and enhances the accuracy and granularity of galaxy classification, facilitating comprehensive studies of cosmic structures and evolution.</p>

        <h3>NLP-Powered Object Description for Astronomical Images</h3>
        <p>This tool leverages advanced Natural Language Processing (NLP) techniques and the
        Multi-Task Learning (MTL) method developed above, to analyze word descriptions across various tasks simultaneously,
        generating prompts tailored to individual needs. Whether you're a writer seeking inspiration,
        a student preparing for presentations, or a professional brainstorming ideas, this application
        adapts seamlessly to provide relevant and engaging prompts. With its user-friendly interface and
        sophisticated algorithms, this application empowers users to unlock their full creative potential
        effortlessly. Experience the future of prompt generation and redefine the way you approach your
        creative endeavors with this powerful tool.</p>


    </div>
    <div class="column">
        <img src="https://raw.githubusercontent.com/Harry45/azml/main/projects/GalaxyGenius/paper-images/tree.png" width="420" alt="Application Image">
    </div>
</div>

</body>
</html>
"""
