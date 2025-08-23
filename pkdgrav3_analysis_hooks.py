def printf(msg):
    import sys
    sys.stdout.write(msg + '\n')
    sys.stdout.flush()

class MassGrid:
    grid = 0
    order = None
    
    def __init__(self, grid, order=None):

        from PKDGRAV import ASSIGNMENT_ORDER

        if order is None:
            order = ASSIGNMENT_ORDER.PCS
    
        self.grid = grid
        self.order = order
    
    def __call__(self, msr, step, time, **kwargs):
    
        print('calculating density grid')
        msr.grid_create(self.grid)
        msr.assign_mass(order=self.order)
        msr.grid_write('output.{:05d}'.format(step))
        msr.grid_delete()
    
    def ephemeral(self, msr, **kwargs):
    
        return msr.grid_ephemeral(self.grid)


class HealpixConverter:

    def __init__(self, namespace, nside):

        self.namespace = namespace
        self.nside = nside

    def __call__(self, msr, step, time, **kwargs):

        if step == 0:
            printf('Merging healpix maps: step: {}, skipping'.format(step))
            return

        import os, glob, h5py
        import healpy as hp
        import numpy as np

        # Find the hpb files
        id_hpb = step - 1
        pattern = "{}.{:05d}.hpb.*".format(self.namespace, id_hpb)
        list_hpb_files = glob.glob(pattern)

        if len(list_hpb_files) == 0:
            printf('Merging healpix maps: step: {}, no hpb files found'.format(id_hpb))
            return

        # Read the hpb files
        num_pix = hp.nside2npix(self.nside)
        current_pixels = []
        for i in range(len(list_hpb_files)):

            f = "{}.{:05d}.hpb.{:d}".format(self.namespace, id_hpb, i)
            data = np.fromfile(f, count=-1, dtype=np.uint32)
            current_pixels.append(data[0::3] + data[1::3])
    
        current_pixels = np.concatenate(current_pixels)
        assert np.sum(current_pixels[num_pix:]) == 0, "Merging healpix maps: step: {}, truncation did not work (pkd_collect)".format(id_hpb)
        current_pixels = current_pixels[:num_pix]

        # Write the hpb file
        fname_out = "{}.hpb.{:05d}.h5".format(self.namespace, id_hpb)
        with h5py.File(fname_out, 'w') as f:
            f.create_dataset(name='hpb', data=current_pixels, compression='lzf', shuffle=True)
            f['hpb'].attrs['nside'] = int(self.nside)
            f['hpb'].attrs['npix'] = int(num_pix)
            f['hpb'].attrs['step'] = int(id_hpb)


        # Remove the hpb files
        # for f in list_hpb_files:
        #     if os.path.isfile(f):
        #         os.remove(f)

        printf('Merging healpix maps: step: {}, stored in: {}'.format(id_hpb, fname_out))


class LightconeConverter:

    def __init__(self, namespace, output_format='h5'):

        self.namespace = namespace
        self.output_format = output_format

    def __call__(self, msr, step, time, a, **kwargs):

        if step == 0:
            printf('Merging lightcone particles: step: {}, skipping'.format(step))
            return

        import os, glob, h5py
        import healpy as hp
        import numpy as np

        # Find the lcp files
        id_hpb = step - 1
        pattern = "{}.{:05d}.lcp.*".format(self.namespace, id_hpb)
        list_lcp_files = glob.glob(pattern)
        printf(f'{len(list_lcp_files):>6d} lcp files found')

        if len(list_lcp_files) == 0:
            printf('Merging lightcone particles: step: {}, no lcp files found'.format(id_hpb))
            return

        # Read the lcp files
        current_particles = []
        for i in range(len(list_lcp_files)):

            f = "{}.{:05d}.lcp.{:d}".format(self.namespace, id_hpb, i)
            data = read_pkdgrav3_particles(f, id_dtype='u8', endian='<', mmap=False)
            if len(data) == 0:
                continue

            current_particles.append(data)
            printf(f'{f:<60s} n={data.shape[0]:<10d} min={np.min(data["pos"]): 10.4e} max={np.max(data["pos"]): 10.3e}')

        # Concatenate the particles
        if len(current_particles) > 0:
            current_particles = np.concatenate(current_particles)
        else:
            current_particles = np.empty(0, dtype=np.dtype([('pid', 'u8'), ('pos', 'f4', (3,)), ('vel', 'f4', (3,)), ('pot', 'f4')]))

        # Write the lcp file
        if self.output_format == 'h5':

            fname_out = "{}.lcp.{:05d}.h5".format(self.namespace, id_hpb)
            with h5py.File(fname_out, 'w') as f:
                f.create_dataset(name='lcp', data=current_particles, compression='lzf', shuffle=True)
                f['lcp'].attrs['step'] = int(id_hpb)
                f['lcp'].attrs['a'] = a

        elif self.output_format == 'gadget4hdf5':

            fname_out = "{}.lcp.{:05d}.gadget4hdf5".format(self.namespace, id_hpb)
            write_gadget4_hdf5(fname_out, current_particles, a=a, boxsize=1.0, mass=1.0, compression='gzip', compression_opts=1)


        printf('Merging lightcone particles: step: {}, num_particles: {}, format: {}, stored in: {}'.format(id_hpb, len(current_particles), self.output_format, fname_out))

        # Remove the lcp files
        # for f in list_lcp_files:
        #     if os.path.isfile(f):
        #         os.remove(f)




import numpy as np
from pathlib import Path

_RECORD_BYTES = 40

def read_pkdgrav3_particles(path, *, id_dtype='u8', endian='<', mmap=False):
    """
    Read a PKDGRAV3 lightcone particle binary file into a NumPy structured array.

    Record layout per particle (40 bytes total):
      0:  64-bit integer   -> particle ID
      8:  3 * float32      -> position  (x, y, z)
      20: 3 * float32      -> velocity  (vx, vy, vz)
      32: float32          -> potential
      36: 4 bytes          -> padding (ignored)

    Parameters
    ----------
    path : str | Path
        Path to the binary particle file.
    id_dtype : str | np.dtype, optional
        Dtype for particle ID: 'u8' (uint64) or 'i8' (int64). Default 'u8'.
    endian : {'<','>','='}, optional
        Byte order of the file: little '<' (default), big '>', or native '='.
    mmap : bool, optional
        If True, return a read-only np.memmap (zero-copy). Default False.

    Returns
    -------
    np.ndarray
        Structured array with fields:
          - 'pid' : uint64/int64
          - 'pos' : float32[3]
          - 'vel' : float32[3]
          - 'pot' : float32
        The final 4 bytes of padding per record are skipped automatically.
        Empty files return an empty array with the same dtype.

    Raises
    ------
    ValueError
        If file size is not a multiple of 40 bytes.
    """
    path = Path(path)
    nbytes = path.stat().st_size

    # Build dtype that matches on-disk layout and ignores the 4 padding bytes
    pid_dt = np.dtype(id_dtype).newbyteorder(endian)
    f4 = np.dtype('f4').newbyteorder(endian)
    dtype = np.dtype({
        'names':   ['pid',      'pos',        'vel',        'pot'],
        'formats': [ pid_dt,     (f4, (3,)),   (f4, (3,)),   f4   ],
        'offsets': [ 0,          8,            20,           32   ],
        'itemsize': _RECORD_BYTES
    })

    if nbytes == 0:
        return np.empty(0, dtype=dtype)

    if nbytes % _RECORD_BYTES != 0:
        raise ValueError(f"File size ({nbytes} bytes) is not a multiple of {_RECORD_BYTES}.")

    count = nbytes // _RECORD_BYTES

    if mmap:
        return np.memmap(path, dtype=dtype, mode='r', shape=(count,))
    else:
        with open(path, 'rb') as f:            
            arr = np.fromfile(f, dtype=dtype, count=-1)

        return arr







import numpy as np
import h5py
from pathlib import Path

def write_gadget4_hdf5(
    path,
    particles,
    *,
    parttype=1,                 # 1 = DM, 0 = gas, etc.
    a=1.0,                      # scale factor to store in Header['Time']
    boxsize=0.0,                # comoving box size in code units (0 if not periodic)
    ntypes=6,                   # length of header arrays (usually 6)
    mass=None,                  # constant mass per particle (code units)
    masses=None,                # or per-particle masses (length N, code units)
    vel_already_u=True,         # True if velocities are already Gadget-u; False if they are physical v
    num_files_per_snapshot=1,
    header_extras=None,         # dict of extra Header attributes (e.g. Omega0, HubbleParam) for compatibility
    units=None,                 # dict with e.g. {'UnitLength_in_cm':..., 'UnitMass_in_g':..., 'UnitVelocity_in_cm_per_s':..., 'HubbleParam':..., ...}
    compression=None,           # e.g. 'gzip' (or None)
    compression_opts=4,         # gzip level if compression='gzip'
    dtype_pos=np.float32,
    dtype_vel=np.float32,
    dtype_pot=np.float32,
    link_particletype_alias=True  # also create 'ParticleTypeX' soft link for compatibility
):
    """
    Write a single-file GADGET-4 HDF5 snapshot containing one PartType group.

    Parameters
    ----------
    path : str | Path
        Output HDF5 filename (e.g. 'snapshot_000.hdf5').
    particles : np.ndarray
        Structured array with fields:
            'pid' : uint64/int64, shape (N,)
            'pos' : float,        shape (N, 3)
            'vel' : float,        shape (N, 3)
            'pot' : float,        shape (N,)     [optional]
    parttype : int
        Gadget particle type to write into (0..ntypes-1). DM is typically 1.
    a : float
        Value stored in Header['Time']. For cosmology, this is the scale factor.
    boxsize : float
        Comoving box size in internal length units (0.0 if not periodic).
    mass, masses :
        Supply either a constant mass per particle (`mass`), or a 1D array (`masses`).
        Gadget requires MassTable[type] > 0 for constant mass, or Masses dataset otherwise.
    vel_already_u : bool
        If False, converts input peculiar v to Gadget-u via u = v / sqrt(a).
    header_extras : dict
        Additional Header attributes for tool compatibility (e.g., 'Omega0', 'OmegaLambda', 'HubbleParam').
    units : dict
        If provided, will be added to a 'Parameters' group (and mirrored into Header if keys match),
        e.g. {'UnitLength_in_cm': 3.08567758e21, 'UnitVelocity_in_cm_per_s': 1e5, 'UnitMass_in_g': 1.98847e33, 'HubbleParam': 0.6774}.
        We also attach minimalist, Arepo/TNG-style per-dataset unit hints when possible.
    """
    path = Path(path)
    N = int(len(particles))

    # Pull arrays & cast dtypes
    pos = np.asarray(particles['pos'], dtype=dtype_pos, order='C')
    vel = np.asarray(particles['vel'], dtype=dtype_vel, order='C')
    if not vel_already_u:
        if a <= 0:
            raise ValueError("a (scale factor) must be > 0 when converting v -> u.")
        vel = vel / np.sqrt(a)
    pids = np.asarray(particles['pid'])  # preserve integer dtype
    pot = np.asarray(particles['pot'], dtype=dtype_pot) if ('pot' in particles.dtype.names) else None

    # Mass handling per Gadget rules
    mass_table = np.zeros(ntypes, dtype=np.float64)
    if masses is not None:
        masses = np.asarray(masses)
        if masses.shape != (N,):
            raise ValueError(f"`masses` must have shape ({N},)")
        mass_table[parttype] = 0.0  # per-particle masses written
    elif mass is not None:
        mass_table[parttype] = float(mass)
        masses = None
    else:
        raise ValueError(
            "Provide either `mass` (constant) or `masses` (per-particle). "
            "Gadget requires MassTable[type] > 0 or a Masses dataset."
        )

    # Header arrays: counts per type
    num_this = np.zeros(ntypes, dtype=np.uint32)
    num_tot  = np.zeros(ntypes, dtype=np.uint64)
    num_this[parttype] = N
    num_tot[parttype]  = N

    # Open file and write
    with h5py.File(path, "w") as f:
        # --- Header (attributes only) ---
        h = f.create_group("Header")
        h.attrs["NumPart_ThisFile"]   = num_this
        h.attrs["NumPart_Total"]      = num_tot
        h.attrs["MassTable"]          = mass_table
        h.attrs["Time"]               = float(a)
        h.attrs["Redshift"]           = float(1.0 / a - 1.0) if a > 0 else 0.0
        h.attrs["BoxSize"]            = float(boxsize)
        h.attrs["NumFilesPerSnapshot"]= int(num_files_per_snapshot)
        # Optional extras for tool compatibility (older readers may expect these)
        if header_extras:
            for k, v in header_extras.items():
                h.attrs[k] = v

        # Optional 'Parameters' group with unit info, hubble param, etc.
        if units:
            p = f.create_group("Parameters")
            for k, v in units.items():
                p.attrs[k] = v
                # Mirror a few common ones into Header for compatibility with some tools
                if k in ("HubbleParam", "Omega0", "OmegaLambda", "OmegaBaryon",
                         "UnitLength_in_cm", "UnitMass_in_g", "UnitVelocity_in_cm_per_s"):
                    h.attrs[k] = v

        # --- Particle group ---
        gname = f"PartType{parttype}"
        g = f.create_group(gname)

        # Convenience to write datasets with optional attributes
        def _write(name, data, *, cgs=None, aexp=None, hexp=None):
            ds = g.create_dataset(
                name, data=data,
                compression=compression, compression_opts=compression_opts,
                shuffle=(compression is not None)
            )
            # Attach minimal unit hints if provided
            if cgs is not None:
                ds.attrs["CGSConversionFactor"] = float(cgs)
            if aexp is not None:
                ds.attrs["aexp-scale-exponent"] = float(aexp)
            if hexp is not None:
                ds.attrs["h-scale-exponent"] = float(hexp)
            return ds

        # Pull unit conversion factors if supplied
        L_cgs = units.get("UnitLength_in_cm")           if isinstance(units, dict) else None
        V_cgs = units.get("UnitVelocity_in_cm_per_s")   if isinstance(units, dict) else None
        M_cgs = units.get("UnitMass_in_g")              if isinstance(units, dict) else None

        # Coordinates, Velocities, IDs (required blocks for collisionless particles)
        # (GADGET-4 HDF5 uses dataset names 'Coordinates', 'Velocities', 'ParticleIDs'.)
        _write("Coordinates", pos, cgs=L_cgs, aexp=1.0 if L_cgs else None, hexp=-1.0 if L_cgs else None)
        _write("Velocities",  vel, cgs=V_cgs, aexp=0.5 if V_cgs else None,  hexp=None)
        g.create_dataset("ParticleIDs", data=pids,
                         compression=compression, compression_opts=compression_opts,
                         shuffle=(compression is not None))

        # Masses (only if variable masses requested)
        if masses is not None:
            _write("Masses", masses, cgs=M_cgs, aexp=0.0 if M_cgs else None, hexp=-1.0 if M_cgs else None)

        # Potential (optional in GADGET; include if present in your array)
        if pot is not None:
            _write("Potential", pot,
                   cgs=(V_cgs * V_cgs) if V_cgs else None,
                   aexp=-1.0 if V_cgs else None,
                   hexp=None)

        # Create a compatibility alias 'ParticleTypeX' -> 'PartTypeX' (soft link)
        if link_particletype_alias:
            f[f"ParticleType{parttype}"] = h5py.SoftLink("/" + gname)


















def test_lightcone_converter(namespace, step):

    msr = None
    lightcone_converter = LightconeConverter(namespace=namespace, output_format='gadget4hdf5')
    lightcone_converter(msr, step=step, time=0, a=1.0)

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description="Test pkdgrav3 analysis hooks.")
    parser.add_argument('--namespace', type=str, default='CosmoGridV2bench', help='Namespace for output files')
    parser.add_argument('--step', type=int, default=0, help='Step number')
    args = parser.parse_args()

    test_lightcone_converter(args.namespace, args.step)
