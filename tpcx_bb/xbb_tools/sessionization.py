#
# Copyright (c) 2019-2020, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import cupy as cp
import numpy as np
from numba import cuda

from nvtx import annotate
from nvtx import start_range
from nvtx import end_range

def get_session_id_from_session_boundry(session_change_df, last_session_len):
    """
        This function returns session starts given a session change df
    """
    import cudf

    ## we dont really need the `session_id` to start from 0
    ## the total number of sessions per partition should be fairly limited
    ## and we really should not hit 2,147,483,647 sessions per partition
    ## Can switch to vec_arange code to match spark 1-1
    ## see previously commited code
    ## https://github.com/rapidsai/tpcx-bb/blob/8394f2b8d62540b4077c606c8b687dee96b4f5d3/tpcx-bb1.3.1/tools/sessionization.py

    x = start_range("arrange", "cyan", "cudf_python")
    user_session_ids = cp.arange(len(session_change_df), dtype=np.int32)
    end_range(x)

    ### up shift the session length df
    x = start_range("shift_and_diff", "green", "cudf_python")
    session_len = session_change_df["t_index"].diff().reset_index(drop=True)
    session_len = session_len.shift(-1)
    session_len.iloc[-1] = last_session_len
    end_range(x)

    x = start_range("repeat", "red", "cudf_python")
    session_id_final_series = (
        cudf.Series(user_session_ids).repeat(session_len).reset_index(drop=True)
    )
    end_range(x)
    return session_id_final_series


@annotate("get_session_id_without_udf", color="orange", domain="cudf_python")
def get_session_id_without_udf(df, keep_cols, time_out):
    """
        This function creates a session id column for each click
        The session id grows in incremeant for each user's susbequent session
        Session boundry is defined by the time_out 
    """
    x = start_range("user_change_flag", "yellow", "cudf_python")
    df["user_change_flag"] = df["wcs_user_sk"].diff(periods=1) != 0
    end_range(x)
    x = start_range("time_delta", "green", "cudf_python")
    df["time_delta"] = df["tstamp_inSec"].diff(periods=1)
    end_range(x)
    x = start_range("session_timeout_flag", "blue", "cudf_python")
    df["session_timeout_flag"] = df["tstamp_inSec"].diff(periods=1) > time_out
    end_range(x)

    x = start_range("session_change_flag", "purple", "cudf_python")
    df["session_change_flag"] = df["session_timeout_flag"] | df["user_change_flag"]
    end_range(x)

    # print(f"Total session change = {df['session_change_flag'].sum():,}")
    x = start_range("select_columns", "green", "cudf_python")
    keep_cols = list(keep_cols)
    keep_cols += ["session_change_flag"]
    df = df[keep_cols]
    end_range(x)

    x = start_range("arrange", "cyan", "cudf_python")
    df = df.reset_index(drop=True)
    df["t_index"] = cp.arange(start=0, stop=len(df), dtype=np.int32)
    end_range(x)

    x = start_range("filter", "red", "cudf_python")
    session_change_df = df[df["session_change_flag"]].reset_index(drop=True)
    last_session_len = len(df) - session_change_df["t_index"].iloc[-1]
    end_range(x)

    session_ids = get_session_id_from_session_boundry(
        session_change_df, last_session_len
    )

    assert len(session_ids) == len(df)
    return session_ids


@annotate("get_session_id_with_udf", color="orange", domain="cudf_python")
def get_session_id_with_udf(df, keep_cols, time_out):
    """
        This function creates a session id column for each click
        The session id grows in incremeant for each user's susbequent session
        Session boundry is defined by the time_out 
    """

      # Launch configuration for all kernels
    NTHRD = 1024
    NBLCK = int(np.ceil(float(len(df)) / float(NTHRD)))
    
    # Preallocate destination column for Numba
    df['session_change_flag'] = cp.zeros(len(df), dtype='int32')
    
    x = start_range("wcs_user_sk", "red", "cudf_python")
    wcs_user_sk = df['wcs_user_sk']._column.data_array_view
    end_range(x)
    x = start_range("tstamp_inSec", "yellow", "cudf_python")
    tstamp_inSec = df['tstamp_inSec']._column.data_array_view
    end_range(x)
    x = start_range("session_change_flag", "green", "cudf_python")
    session_change_flag = df['session_change_flag']._column.data_array_view
    end_range(x)
    
    # Determine the session boundaries
    x = start_range("session_change_flag_kernel", "blue", "cudf_python")
    make_session_change_flag_kernel[NBLCK, NTHRD](wcs_user_sk, 
                                                 tstamp_inSec, 
                                                 session_change_flag, 
                                                 time_out)
    end_range(x)
    
    x = start_range("population_kernel", "blue", "cudf_python")
    populate_session_ids_kernel[NBLCK, NTHRD](session_change_flag)    
    end_range(x)

    return df['session_change_flag']


def get_sessions(df, with_udf, keep_cols, time_out=3600):
    df = df.sort_values(by=["wcs_user_sk", "tstamp_inSec"]).reset_index(drop=True)
    if with_udf:
        df["session_id"] = get_session_id_with_udf(df, keep_cols, time_out)
    else:
        df["session_id"] = get_session_id_without_udf(df, keep_cols, time_out)
    keep_cols += ["session_id"]
    df = df[keep_cols]
    return df


def get_distinct_sessions(df, with_udf, keep_cols, time_out=3600):
    """
        ### Performence note
        The session + distinct 
        logic takes 0.2 seconds for a dataframe with 10M rows
        on gv-100
    """
    df = get_sessions(df, with_udf, keep_cols, time_out=3600)
    df = df.drop_duplicates().reset_index(drop=True)
    return df


def get_pairs(
    df,
    merge_col=["session_id", "wcs_user_sk"],
    pair_col="i_category_id",
    output_col_1="category_id_1",
    output_col_2="category_id_2",
):
    """
        Gets pair after doing a inner merge
    """
    pair_df = df.merge(df, on=merge_col, suffixes=["_t1", "_t2"], how="inner")
    pair_df = pair_df[[f"{pair_col}_t1", f"{pair_col}_t2"]]
    pair_df = pair_df[
        pair_df[f"{pair_col}_t1"] < pair_df[f"{pair_col}_t2"]
    ].reset_index(drop=True)
    pair_df.columns = [output_col_1, output_col_2]
    return pair_df

@cuda.jit
def make_session_change_flag_kernel(wcs_user_sk, tstamp_inSec, session_change_flag, time_out):
    gid = cuda.threadIdx.x + (cuda.blockIdx.x * cuda.blockDim.x)
    if 0 < gid < len(wcs_user_sk):
        if wcs_user_sk[gid] != wcs_user_sk[gid-1] or tstamp_inSec[gid] - tstamp_inSec[gid-1] > time_out:
            session_change_flag[gid] = np.int32(1)
        else:
            session_change_flag[gid] = np.int32(0)
    else:
        session_change_flag[gid] = np.int32(1)
        

@cuda.jit
def populate_session_ids_kernel(session_boundary):
        gid = cuda.threadIdx.x + (cuda.blockIdx.x * cuda.blockDim.x)
        # don't loop if we're off the edge
        if gid < len(session_boundary):
                # if this is a session boundary...
                if session_boundary[gid] == 1:
                        # this thread marks the start of a session
                        session_boundary[gid] = gid
                        look_ahead = 1

                        # check elements 'forward' of this one 
                        # until a new session boundary is found
                        while session_boundary[gid + look_ahead] == 0:
                                session_boundary[gid + look_ahead] = gid
                                look_ahead += 1

                                # don't segfault if I'm the last thread
                                if gid + look_ahead == len(session_boundary) - 1:
                                        break
