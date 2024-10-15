# -*- coding: utf-8 -*-
import matplotlib
import matplotlib.pyplot as plt
from hengck.data import *
level_to_label={
    'l1_l2':1,
    'l2_l3':2,
    'l3_l4':3,
    'l4_l5':4,
    'l5_s1':5,
}

#################################################

# study id used for demo
id_df = pd.read_csv(f'{DATA_KAGGLE_DIR}/train_series_descriptions.csv')
valid_id = [ 113758629, 13317052, 60612428, 74294498, 142991438,
            168833126, 189360935, 58813022, 1115952008, 959290081,
           ] #these are not used in training
valid_id = [11943292]
image_size = 512

for study_id in valid_id:
    # d = id_df[(id_df.study_id == study_id)].iloc[0]
    axial_t2_id = id_df[(id_df.study_id == study_id) & (id_df.series_description == 'Axial T2')].iloc[0].series_id
    sagittal_t2_id = id_df[(id_df.study_id == study_id) & (id_df.series_description == 'Sagittal T2/STIR')].iloc[
        0].series_id

    axial_t2_id = 3800798510
    data = read_study(study_id, axial_t2_id=axial_t2_id, sagittal_t2_id=sagittal_t2_id)

    # --- step.1 : detect 2d point in sagittal_t2
    sagittal_t2 = data.sagittal_t2.volume
    sagittal_t2_df = data.sagittal_t2.df
    axial_t2_df = data.axial_t2.df


    D, H, W = sagittal_t2.shape
    image = resize_volume(sagittal_t2, image_size)

    sagittal_t2_z = D // 2
    image = image[sagittal_t2_z]  # we use only center image #todo: better selection

   
    # for debug and development
    point_hat, z_hat = sagittal_t2_point_hat = get_true_sagittal_t2_point(study_id, sagittal_t2_df)
    point_hat = point_hat * [[image_size / W, image_size / H]]

    sagittal_t2_point = point_hat
    # --- step.2 : perdict slice level of axial_t2
    world_point = view_to_world(sagittal_t2_point, sagittal_t2_z, sagittal_t2_df,
                                              image_size)
    assigned_level, closest_z, dis = axial_t2_level = point_to_level(world_point, axial_t2_df)
    print('assigned_level:', assigned_level)
    print('closest_z: ', closest_z)
    #break

    ###################################################################
    # visualisation

    # https://matplotlib.org/stable/gallery/mplot3d/mixed_subplots.html
    fig = plt.figure(figsize=(23, 6))
    ax1 = fig.add_subplot(1, 1, 1)
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')

    print(point_hat.shape)
    print(point_hat)
    # detection result
    probability = np.zeros((6, image_size, image_size), dtype=np.uint8)
    for i, pt in enumerate(point_hat):
        x, y = int(pt[0]), int(pt[1])
        cv2.circle(probability[i+1], (x, y), 7, (255, 255, 255), 5)
    probability = probability.astype(np.float32) / 255

    p = probability_to_rgb(probability)
    m = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    m = 255 - (255 - m * 0.8) * (1 - p / 255)

    ax1.imshow(m / 255)
    ax1.set_title(f'sagittal keypoint detection (unet)\n series_id: {axial_t2_id}')

    # draw  assigned_level
    level_ncolor = np.array(level_color) / 255
    coloring = level_ncolor[assigned_level].tolist()
    draw_slice(
        ax2, axial_t2_df,
        is_slice=True, scolor=coloring, salpha=[0.1],
        is_border=True, bcolor=coloring, balpha=[0.2],
        is_origin=False, ocolor=[[0, 0, 0]], oalpha=[0.0],
        is_arrow=True
    )

    # draw world_point
    ax2.scatter(world_point[:, 0], world_point[:, 1], world_point[:, 2], alpha=1, color='black')

    ### draw closest slice
    coloring = level_ncolor[1:].tolist()
    draw_slice(
        ax2, axial_t2_df.iloc[closest_z],
        is_slice=True, scolor=coloring, salpha=[0.1],
        is_border=True, bcolor=coloring, balpha=[1],
        is_origin=False, ocolor=[[1, 0, 0]], oalpha=[0],
        is_arrow=False
    )

    ax2.set_aspect('equal')
    ax2.set_title(f'axial slice assignment\n series_id:{sagittal_t2_id}')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('z')
    ax2.view_init(elev=0, azim=-10, roll=0)
    plt.tight_layout(pad=2)
    plt.show()
    break