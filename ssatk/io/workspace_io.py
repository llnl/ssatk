import json
import inspect
import numpy as np
from ssatk._paths import ensure_file_parent
from datetime import datetime
from pathlib import Path

# Optional imports - will handle gracefully if not installed
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import astropy.units as u
    from astropy.table import Table
    from astropy.time import Time
    from astropy.coordinates import SkyCoord
    HAS_ASTROPY = True
except ImportError:
    HAS_ASTROPY = False


def _representation_type_name(representation_type):
    name = getattr(representation_type, "name", None)
    if name is not None:
        return name

    return str(representation_type)


def save_workspace(filename='workspace.json', exclude=None):
    """
    Save all variables from the current workspace to a JSON file.
    
    Parameters:
    -----------
    filename : str, optional
        Name of the file to save to (default: 'workspace.json')
    exclude : list, optional
        List of variable names to exclude from saving
    
    Returns:
    --------
    dict : Dictionary containing saved variables and metadata
    """
    # Get the caller's frame to access their variables
    caller_frame = inspect.currentframe().f_back
    caller_globals = caller_frame.f_globals
    caller_locals = caller_frame.f_locals
    
    # Combine globals and locals (locals take precedence)
    all_vars = {**caller_globals, **caller_locals}
    
    # Initialize exclude list
    if exclude is None:
        exclude = []
    
    # Variables to automatically exclude
    auto_exclude = {
        '__name__', '__doc__', '__package__', '__loader__', '__spec__',
        '__annotations__', '__builtins__', '__file__', '__cached__'
    }
    exclude_set = auto_exclude.union(set(exclude))
    
    # Dictionary to store serializable variables
    workspace = {}
    skipped = []
    
    for name, value in all_vars.items():
        # Skip excluded variables, functions, modules, and classes
        if (name in exclude_set or 
            name.startswith('_') or
            inspect.ismodule(value) or 
            inspect.isfunction(value) or
            inspect.isclass(value) or
            inspect.isbuiltin(value)):
            continue
        
        try:
            # Handle different types
            if HAS_ASTROPY and isinstance(value, u.Quantity):
                workspace[name] = {
                    '__type__': 'astropy.Quantity',
                    'value': value.value.tolist() if hasattr(value.value, 'tolist') else float(value.value),
                    'unit': str(value.unit)
                }
            elif isinstance(value, np.ndarray):
                workspace[name] = {
                    '__type__': 'numpy.ndarray',
                    'data': value.tolist(),
                    'dtype': str(value.dtype),
                    'shape': value.shape
                }
            elif HAS_PANDAS and isinstance(value, pd.DataFrame):
                workspace[name] = {
                    '__type__': 'pandas.DataFrame',
                    'data': value.to_dict(orient='tight'),
                    'index_name': value.index.name,
                    'columns_name': value.columns.name
                }
            elif HAS_PANDAS and isinstance(value, pd.Series):
                workspace[name] = {
                    '__type__': 'pandas.Series',
                    'data': value.to_dict(),
                    'index': value.index.tolist(),
                    'name': value.name,
                    'dtype': str(value.dtype)
                }
            elif HAS_ASTROPY and isinstance(value, Table):
                workspace[name] = {
                    '__type__': 'astropy.Table',
                    'data': {col: value[col].tolist() for col in value.colnames},
                    'colnames': value.colnames,
                    'meta': dict(value.meta) if value.meta else {}
                }
            elif HAS_ASTROPY and isinstance(value, Time):
                workspace[name] = {
                    '__type__': 'astropy.Time',
                    'value': value.isot,
                    'format': 'isot',
                    'scale': value.scale
                }
            elif HAS_ASTROPY and isinstance(value, SkyCoord):
                workspace[name] = {
                    '__type__': 'astropy.SkyCoord',
                    'ra': value.ra.deg,
                    'dec': value.dec.deg,
                    'frame': value.frame.name,
                    'representation_type': _representation_type_name(value.representation_type)
                }
            elif isinstance(value, (datetime,)):
                workspace[name] = {
                    '__type__': 'datetime',
                    'data': value.isoformat()
                }
            elif isinstance(value, set):
                workspace[name] = {
                    '__type__': 'set',
                    'data': list(value)
                }
            elif isinstance(value, (str, int, float, bool, list, dict, tuple, type(None))):
                # JSON-serializable types
                workspace[name] = value
            else:
                # Try to serialize, but skip if it fails
                json.dumps(value)
                workspace[name] = value
        except (TypeError, ValueError, AttributeError):
            skipped.append(name)
            continue
    
    # Add metadata
    result = {
        '__metadata__': {
            'saved_at': datetime.now().isoformat(),
            'num_variables': len(workspace),
            'skipped_variables': skipped
        },
        'variables': workspace
    }
    
    # Save to file
    path = ensure_file_parent(filename)
    with path.open('w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    
    print(f"✓ Workspace saved to '{filename}'")
    print(f"  - Saved {len(workspace)} variables")
    if skipped:
        print(f"  - Skipped {len(skipped)} non-serializable variables: {', '.join(skipped)}")
    
    return result


def load_workspace(filename='workspace.json', into_globals=True):
    """
    Load variables from a JSON file back into the workspace.
    
    Parameters:
    -----------
    filename : str, optional
        Name of the file to load from (default: 'workspace.json')
    into_globals : bool, optional
        If True, load variables into the caller's global scope (default: True)
        If False, return them as a dictionary
    
    Returns:
    --------
    dict : Dictionary of loaded variables
    """
    # Load from file
    with Path(filename).open('r', encoding='utf-8') as f:
        data = json.load(f)
    
    variables = data.get('variables', {})
    loaded_vars = {}
    
    # Reconstruct special types
    for name, value in variables.items():
        if isinstance(value, dict) and '__type__' in value:
            type_name = value['__type__']
            
            if type_name == 'numpy.ndarray':
                loaded_vars[name] = np.array(value['data'], dtype=value['dtype'])
            elif type_name == 'pandas.DataFrame' and HAS_PANDAS:
                df = pd.DataFrame.from_dict(value['data'], orient='tight')
                if value.get('index_name'):
                    df.index.name = value['index_name']
                if value.get('columns_name'):
                    df.columns.name = value['columns_name']
                loaded_vars[name] = df
            elif type_name == 'pandas.Series' and HAS_PANDAS:
                series_data = value['data']
                if isinstance(series_data, dict):
                    series_data = [
                        series_data.get(idx, series_data.get(str(idx)))
                        for idx in value['index']
                    ]
                loaded_vars[name] = pd.Series(
                    series_data,
                    index=value['index'],
                    name=value.get('name')
                ).astype(value['dtype'])
            elif type_name == 'astropy.Table' and HAS_ASTROPY:
                from astropy.table import Table
                loaded_vars[name] = Table(
                    value['data'],
                    names=value['colnames'],
                    meta=value.get('meta', {})
                )
            elif type_name == 'astropy.Quantity' and HAS_ASTROPY:
                loaded_vars[name] = u.Quantity(value['value'], unit=value['unit'])
            elif type_name == 'astropy.Time' and HAS_ASTROPY:
                from astropy.time import Time
                loaded_vars[name] = Time(value['value'], format=value['format'], scale=value['scale'])
            elif type_name == 'astropy.SkyCoord' and HAS_ASTROPY:
                from astropy.coordinates import SkyCoord
                loaded_vars[name] = SkyCoord(
                    ra=value['ra']*u.deg,
                    dec=value['dec']*u.deg,
                    frame=value['frame']
                )
            elif type_name == 'datetime':
                loaded_vars[name] = datetime.fromisoformat(value['data'])
            elif type_name == 'set':
                loaded_vars[name] = set(value['data'])
        else:
            loaded_vars[name] = value
    
    # Inject into caller's globals if requested
    if into_globals:
        caller_frame = inspect.currentframe().f_back
        caller_globals = caller_frame.f_globals
        caller_globals.update(loaded_vars)
    
    metadata = data.get('__metadata__', {})
    print(f"✓ Workspace loaded from '{filename}'")
    print(f"  - Loaded {len(loaded_vars)} variables")
    if metadata.get('saved_at'):
        print(f"  - Originally saved at: {metadata['saved_at']}")
    
    return loaded_vars


# Example usage
if __name__ == "__main__":
    # Create some example variables
    x = 42
    y = [1, 2, 3, 4, 5]
    name = "test_workspace"
    data_dict = {"a": 1, "b": 2, "c": 3}
    matrix = np.array([[1, 2], [3, 4]])
    
    # Pandas examples (if available)
    if HAS_PANDAS:
        df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': ['x', 'y', 'z'],
            'C': [1.1, 2.2, 3.3]
        })
        series = pd.Series([10, 20, 30], name='my_series')
    
    # Astropy examples (if available)
    if HAS_ASTROPY:
        distance = 5.0 * u.parsec
        velocity = 200 * u.km / u.s
        time = Time('2023-01-01T00:00:00', format='isot', scale='utc')
        coord = SkyCoord(ra=10.68*u.deg, dec=41.27*u.deg, frame='icrs')
        table = Table({'name': ['star1', 'star2'], 'mag': [12.3, 15.6]})
    
    # Save the workspace
    save_workspace('my_workspace.json')
    
    # Clear variables
    del x, y, name, data_dict, matrix
    if HAS_PANDAS:
        del df, series
    if HAS_ASTROPY:
        del distance, velocity, time, coord, table
    
    # Load them back
    load_workspace('my_workspace.json')
    
    # Verify they're restored
    print(f"\nRestored variables:")
    print(f"x = {x}")
    print(f"y = {y}")
    print(f"name = {name}")
    print(f"matrix = \n{matrix}")
    
    if HAS_PANDAS:
        print(f"\nDataFrame:\n{df}")
        print(f"\nSeries:\n{series}")
    
    if HAS_ASTROPY:
        print(f"\ndistance = {distance}")
        print(f"velocity = {velocity}")
        print(f"time = {time}")
        print(f"coord = {coord}")
        print(f"\nTable:\n{table}")
