import pco
import matplotlib.pyplot as plt
import numpy as np
from time import perf_counter
import os
import matplotlib as mpl
mpl.use('TkAgg')


def get_sample_wood(phase_angle):
    sample = ''
    s_chars = ['S', 'A', 'F', 'M']
    s_inf = [int(x) for x in input("Please enter the sample number, wood class, measurement side"
                                   " and the measurement number as integers, e.g. 1 2 1 4: ").split()]
    s_inf[0] = format(s_inf[0], '03d')
    for s_char, s_int in zip(s_chars, s_inf):
        sample = sample + s_char + str(s_int) + '_'
    sample = sample[:-1]
    phase = '_RF' + str(format(int(phase_angle / 1000), '03d'))
    sample = sample + phase
    if len(sample) < 18:
        print('Incorrect sample name!')
        sample = get_sample_wood(phase_angle)
    else:
        print('Current measurement is ', sample)
    if input('Type No if that is not correct and the entry should be repeated: ') == 'No':
        sample = get_sample_wood(phase_angle)
        
    return sample


def get_sample_polymer(phase_angle):
    NoNo_list = ['No', 'no', 'NO', 'nope']
    s_char = 'M'
    sample = input("Please enter the polymer type and the measurement number as a string/integer, e.g. PE 1: ")
    m = 'M' + str(format(int(sample.split(' ')[1]), '03d'))
    sample = str(sample.split(' ')[0]) + '_' + m
    phase = '_RF' + str(format(int(phase_angle / 1000), '03d'))
    sample = sample + phase
    # ask for confirmation
    print('Current measurement is ', sample)
    if input('Type No if that is not correct and the entry should be repeated: ') in NoNo_list:
        sample = get_sample_polymer(phase_angle)
    return sample


def CameraAPI(save_dir, allPhases=False, saveTaps=False, m_object='polymer', phase_angle_increment=5000):
    """
    Function to automatically measure polymers and wood with Rapid-FLIM
    :param phase_angle_increment: Increment to the phase angle if allPhases is true
    :param allPhases: Boolean, if true all starting phase angles from 0° to 170° are measured with the given increment
    :param saveTaps: Boolean, if true taps are saved as well
    :param m_object: polymer or wood
    :param save_dir: directory to save the images into
    :return: None
    """
    with pco.Camera(debuglevel='error', timestamp='off') as cam:
        print("Connected to camera: ", cam.sdk.get_camera_name()['camera name'])
        m_count = 1
        s_count = 1
        run_cnt = 0
        if not allPhases:
            phase_angle = 70000
        elif allPhases:
            phase_angle = 0
        if m_object == 'wood':
            sample = get_sample_wood(phase_angle)
        elif m_object == 'polymer':
            sample = get_sample_polymer(phase_angle)
        else:
            ValueError('Either no measurement type or no allowed type.')
        if not os.path.isdir(save_dir):
            ValueError('The directory to save the images to does not exist.')

        # 'timestamp': 'binary & ascii'
        cam.configuration = {'trigger': 'software trigger', 'exposure time': 0.2}
        cam.flim_configuration = {
            'frequency': 30_000_000,
            'phase_number': 'manual shifting',
            # 'phase_symmetry': 'phase_symmetry',
            # 'phase_order': 'phase_order',
            'tap_select': 'both',
            'source_select': 'intern',
            'output_waveform': 'rectangular',
            # 'asymmetry_correction': 'asymmetry_correction',
            'output_mode': 'default'}

        # cam.sdk.set_trigger_mode('software trigger')
        fc = pco.Flim(**cam.flim_configuration)
        input('Press Enter to start the measurement')
        cam.record(8, mode='fifo')
        cam.sdk.force_trigger()
        cam.wait_for_first_image()

        while True:
            cam.sdk.set_flim_relative_phase(phase_angle)
            meas_time_start = perf_counter()
            images, metas = cam.images(blocksize=cam.flim_stack_size)
            cam.sdk.force_trigger()
            meas_time_stop = perf_counter()
            ni, nid = fc.calculate_rapid_flim(cam, images)
            calc_time_stop = perf_counter()
            max_ni = np.max(ni)
            if max_ni > 0.6:
                p_text = 'The maximum intensity is {} too high, it must be lower than 0.6'.format(max_ni - 0.6)
            if np.max(ni) > 0.6 and run_cnt == 0:
                # check if the intensity is under the set maximum of 0.6 and over 0.4 to avoid dim measurements
                old_exp_time = cam.sdk.get_delay_exposure_time()
                print('Warning! ' + p_text + 'The current exposure time is ' + str(old_exp_time['exposure']) +
                      str(old_exp_time['exposure timebase']))
                exposure_time = input('Input new exposure time in ms: ')
                cam.sdk.set_delay_exposure_time(0, 'ms', int(exposure_time), str(old_exp_time['exposure timebase']))
                run_cnt += 1
                continue
            elif 0 < run_cnt < 2:
                # necessary to run the measurement twice after changing the exposure time else it will take no effect
                run_cnt = 0
                continue

            # flim_plot(ni, nid, phase_angle)
            # meas_time = (meas_time_stop - meas_time_start) * (10**3)
            # calc_time = (calc_time_stop - meas_time_stop) * (10**3)
            # print('Measurement Time: ', meas_time)
            # print('Calculation Time: ', calc_time)
            stack = fc.get_stack(images)
            img = np.array([ni, nid])
            np.save(save_dir + sample, img)
            if saveTaps:
                np.save(save_dir + sample + '_OG', stack)

            if allPhases and phase_angle < 170000:
                phase_angle += phase_angle_increment
                sample = sample[:-3] + str(format(int(phase_angle / 1000), '03d'))
            elif allPhases and phase_angle == 170000:
                # print('Measurement finished at phase angle ', str(int(phase_angle/1000)))
                phase_angle = 0
                if m_object == 'wood':
                    m_count = int(sample.split('_')[3][-1:])
                    if m_count < 4:
                        m_count += 1
                        sample = sample[:-3] + str(format(int(phase_angle / 1000), '03d'))
                        sample = sample[:12] + str(m_count) + sample[13:]
                        print('Current measurement is ', sample)
                        input('Press Enter to start the measurement')
                    else:
                        print('\r', end='')
                        sample = get_sample_wood(phase_angle)
                        input('Press Enter to start the measurement')
                elif m_object == 'polymer':
                    m_count = int(sample.split('_')[1][-1:])
                    if m_count < 50:
                        m_count += 1
                        sample = sample[:-3] + str(format(int(phase_angle / 1000), '03d'))
                        sample = [sample.split('_')]
                        sample[1] = 'M' + str(format(m_count, '03d'))
                        sample = '_'.join(sample)
                        input('Press Enter to start the measurement')
                    else:
                        print('\r', end='')
                        sample = get_sample_polymer(phase_angle)
                        input('Press Enter to start the measurement')

            elif not allPhases:
                if m_object == 'wood':
                    m_count = int(sample.split('_')[3][-1:])
                    if m_count < 4:
                        m_count += 1
                        sample = sample[:12] + str(m_count) + sample[13:]
                        print('Current sample side and measurement is ', sample)
                        input('Press Enter to start the measurement')
                    else:
                        print('\r', end='')
                        sample = get_sample_wood(phase_angle)
                        input('Press Enter to start the measurement')
                elif m_object == 'polymer':
                    m_count = int(sample.split('_')[1][-1:])
                    if m_count < 50:
                        m_count += 1
                        sample = [sample.split('_')]
                        sample[1] = 'M' + str(format(m_count, '03d'))
                        sample = '_'.join(sample)
                        input('Press Enter to start the measurement')
                    else:
                        print('\r', end='')
                        sample = get_sample_polymer(phase_angle)
                        input('Press Enter to start the measurement')
            continue


def flim_plot(ni, nid, phase_angle):
    plt.close('all')
    fig = plt.figure(figsize=(7.33, 4))

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.axes.get_xaxis().set_visible(False)
    ax1.axes.get_yaxis().set_visible(False)
    im1 = plt.imshow(np.asarray(ni), cmap='gray')
    plt.title('Normalized Intensity at ' + str(int(phase_angle/1000)) + '°')
    colorbar(im1)
    plt.clim(np.min(ni), np.max(ni))

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.axes.get_xaxis().set_visible(False)
    ax2.axes.get_yaxis().set_visible(False)
    ax2.set_title('Intensity Difference at ' + str(int(phase_angle/1000)) + '°')
    im2 = plt.imshow(np.asarray(nid), cmap='jet')
    colorbar(im2)
    plt.clim(np.min(nid), np.max(nid))

    plt.subplots_adjust(wspace=0.25, hspace=0.1)
    plt.show(block=False)


def calculate_rapid_flim(self, cam, list_of_images):
    arr = np.array(list_of_images)
    # for i in range(0, 4):
    #     arr = np.delete(arr, i, 1)

    # stack = self.get_stack(list_of_images)

    b = np.mean(arr, axis=0)
    old_exp_time = cam.sdk.get_delay_exposure_time()
    exp_time = old_exp_time['exposure']
    exp_base = old_exp_time['exposure timebase']
    if exp_base == 'ms':
        b = (b * 100) / exp_time
    elif exp_base == 'us':
        exp_time = exp_time * 10 ** (-3)
        b = (b * 100) / exp_time
    elif exp_base == 'ns':
        exp_time = exp_time * 10 ** (-6)
        b = (b * 100) / exp_time
    # b = stack.mean(axis=2)
    bits = 14
    ni = b / ((2 ** bits) - 1)

    c = np.fabs(arr[0, :, :]) - np.fabs(arr[1, :, :])
    e = np.sum(arr, axis=0)
    c = c / ((2 ** bits) - 1)
    e = e / ((2 ** bits) - 1)
    nid = np.nan_to_num((c / e), nan=0.0)
    return ni, nid


def colorbar(mappable):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.pyplot as plt
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    plt.sca(last_axes)
    return cbar


def ImageAquisition():
    with pco.Camera(debuglevel='error', timestamp='off') as cam:
        phase_angle = 70000
        print("Connected to camera: ", cam.sdk.get_camera_name()['camera name'])
        cam.configuration = {'trigger': 'software trigger', 'exposure time': 0.25}
        cam.flim_configuration = {
            'frequency': 30_000_000,
            'phase_number': 'manual shifting',
            'tap_select': 'both',
            'source_select': 'intern',
            'output_waveform': 'rectangular',
            'output_mode': 'default'}

        cam.sdk.set_trigger_mode('software trigger')
        fc = cam.Flim(**cam.flim_configuration)
        while True:
            cam.sdk.set_flim_relative_phase(phase_angle)
            cam.record(8, mode='fifo')
            cam.sdk.force_trigger()
            cam.wait_for_first_image()

            images, metas = cam.images(blocksize=cam.flim_stack_size)
            cam.sdk.force_trigger()
            ni, nid = fc.calculate_rapid_flim(images)
            if np.max(ni) > 0.6:
                old_exp_time = cam.sdk.get_delay_exposure_time()
                exposure_time = round((old_exp_time['exposure'] / 100) * 95)
                cam.sdk.set_delay_exposure_time(0, 'ms', int(exposure_time), str(old_exp_time['exposure timebase']))
                continue
            else:
                return ni, nid
