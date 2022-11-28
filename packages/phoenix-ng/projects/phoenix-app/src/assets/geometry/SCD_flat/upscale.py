def rescale_obj(obj_path, obj_scaled_path, scale):
    with open(obj_path, 'r') as source:
        with open(obj_scaled_path, 'w') as target:
            for line in source:
                taget_line = line

                if(line.startswith('v ')):
                    coordinates = [float(coordinate) for coordinate in line.split(' ')[1:]]
                    rescaled = [c*scale for c in coordinates]
                    rescaled_as_str = " ".join([str(c) for c in rescaled])
                    taget_line = f'v {rescaled_as_str}\n'

                target.write(taget_line)


scale = 2000
sep = 7

rescale_obj('ECAL1_unscaled.obj', 'ECAL1.obj', scale)
rescale_obj('ECAL2_unscaled.obj', 'ECAL2.obj', scale + sep)
rescale_obj('ECAL3_unscaled.obj', 'ECAL3.obj', scale + sep*2)
rescale_obj('HCAL1_unscaled.obj', 'HCAL1.obj', scale + sep*3)
rescale_obj('HCAL2_unscaled.obj', 'HCAL2.obj', scale + sep*4)
rescale_obj('HCAL3_unscaled.obj', 'HCAL3.obj', scale + sep*5)
