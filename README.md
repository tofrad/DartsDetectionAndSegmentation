# Dartsgame Detection and Segmentation

This Repo is about the image-based detection and segmentation of the dartboard and arrows from the darts game. It utilizes transfer learning of a Mask R-CNN Net with a pre-trained ResNet backbone and artificially generated training data.

<table>
  <tr>
    <th colspan="2" style="text-align: center">Example from dataset</th>
  </tr>
  <tr>
    <td style="width: 50%">
      <img src="https://github.com/user-attachments/assets/3a5d89f7-a8a8-4a9b-bf4e-81dc678ffeea" style="width: 100%"/>
    </td>
    <td style="width: 50%; vertical-align: top">
      <img src="https://github.com/user-attachments/assets/62939334-6f8e-40cd-b4f1-38c98bdecb4b" style="width: 100%"/>
    </td>
  </tr>
</table>


The segmented images were filtered for color and polygon masks generated for the COCO dataset format.

<table>
  <tr>
    <th colspan="1" style="text-align: center">Example after dataset creation</th>
  </tr>
  <tr>
    <td style="width: 100%">
      <img width="1563" height="327" alt="grafik" src="https://github.com/user-attachments/assets/7197af83-791f-448b-ba8c-a30f564e6ab9" />
    </td>
  </tr>
</table>

The backbone of the Mask R-CNN is a ResNet-50, pretrained on the COCO dataset images. 

