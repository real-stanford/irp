from typing import Sequence, Tuple

import numpy as np
import mujoco_py as mjp

def get_rope_body_ids(
        model: mjp.cymj.PyMjModel, 
        prefix='B',
        check_topology=True) -> np.ndarray:

    rope_body_names = list()
    name_idx_map = dict()
    for body_name in model.body_names:
        if body_name.startswith(prefix):
            rope_body_names.append(body_name)
            name_idx_map[body_name] = int(body_name.strip(prefix))
    
    rope_body_names = sorted(rope_body_names, key=lambda x: name_idx_map[x])
    rope_body_ids = [model.body_name2id(x) for x in rope_body_names]

    if check_topology:
        for i in range(1, len(rope_body_ids)):
            assert(model.body_parentid[rope_body_ids[i]] \
                == rope_body_ids[i-1])
    return np.array(rope_body_ids)

def get_cloth_body_ids(
        model: mjp.cymj.PyMjModel,
        prefix='B') -> np.ndarray:
    name_idx_map = dict()
    imax = 0
    jmax = 0
    for body_name in model.body_names:
        if body_name.startswith(prefix):
            i, j = [int(x) for x in body_name.strip(prefix).split('_')]
            name_idx_map[body_name] = (i,j)
            imax = max(imax, i)
            jmax = max(jmax, j)
    
    id_arr = np.zeros((imax+1, jmax+1), dtype=np.int64)
    for key, value in name_idx_map.items():
        body_id = model.body_name2id(key)
        id_arr[value] = body_id
    return id_arr

def get_body_center_of_mass(
        data: mjp.cymj.PyMjData,
        body_ids: np.ndarray
        ) -> np.ndarray:
    return data.xipos[body_ids]

def apply_force_com(
        model: mjp.cymj.PyMjModel, 
        data: mjp.cymj.PyMjData,
        body_id: int,
        force: np.ndarray):
    com_point = data.xipos[body_id]
    torque = np.zeros(3)
    mjp.functions.mj_applyFT(
        model, data, 
        force, torque, com_point, 
        body_id, data.qfrc_applied)

def apply_force_com_batch(
        model: mjp.cymj.PyMjModel, 
        data: mjp.cymj.PyMjData,
        body_ids: Sequence[int],
        forces: np.ndarray):
    assert(len(body_ids) == len(forces))
    for i in range(len(body_ids)):
        apply_force_com(model, data, body_ids[i], forces[i])

def clear_forces(data):
    data.qfrc_applied[:] = 0

def apply_impulse_com_batch(
        sim: mjp.cymj.MjSim,
        body_ids: Sequence[int],
        impulses: np.ndarray):
    
    dt = sim.model.opt.timestep
    forces = impulses / dt
    apply_force_com_batch(
        model=sim.model, data=sim.data,
        body_ids=body_ids,
        forces=forces)
    sim.step()
    clear_forces(sim.data)

def get_rope_dof_idxs(
        model: mjp.cymj.PyMjModel, 
        prefix='B',
        check_topology=True
        ) -> Tuple[np.ndarray, np.ndarray]:
    rope_body_ids = get_rope_body_ids(
        model, prefix, check_topology)
    
    dof_body_ids = list()
    dof_idxs = list()
    for body_id in rope_body_ids:
        num_dof = model.body_dofnum[body_id]
        if num_dof == 0:
            continue
        assert(num_dof == 2)
        dof_adr = model.body_dofadr[body_id]
        idxs = [dof_adr + i for i in range(num_dof)]
        dof_idxs.append(idxs)
        dof_body_ids.append(body_id)
    return np.array(dof_idxs), np.array(dof_body_ids)

def get_mujoco_state(sim):
    q = np.copy(sim.data.qpos)
    dq = np.copy(sim.data.qvel)
    u = np.copy(sim.data.ctrl)
    return (q, dq, u)

def set_mujoco_state(sim, state):
    q, dq, u = state
    sim.data.qpos[:] = q
    sim.data.qvel[:] = dq
    sim.data.ctrl[:] = u
