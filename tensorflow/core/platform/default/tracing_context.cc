/*
 * tracing_context.cc
 *
 *  Created on: 2016年11月22日
 *      Author: Bowen Yu <yubowen15@foxmail.com>
 */

#include "tracing_context.h"
#include "logging.h"
#include <unistd.h>
#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>

#include <sstream>


namespace tensorflow {
namespace internal {

#define WRITE_ITEM(TYPE, NAME) do { *((TYPE*)(_buf + offset)) = (NAME); offset += sizeof(TYPE); } while(0)
#define MAX_NUM_THREADS 1024

struct alignas(64) Serializer {
  char* _buf;
  std::ofstream* _fp;
  string _file_path;
  size_t offset;

  Serializer() : _buf(NULL), _fp(NULL), offset(0) {
  }

  virtual ~Serializer() {
    if(_buf != NULL) {
      FILE* fp;
      fp = fopen(_file_path.c_str(), "wb");
      fwrite(_buf, sizeof(char), offset, fp);
      fclose(fp);

      _fp->close();
      delete _fp;
      _fp = NULL;
      LOG(INFO) << "Successfully written to " << _file_path;
    }
  }

  void open(const char* file_path, size_t size) {
    _buf = new char[size];
    _file_path = file_path;
  }

  void write(double timestamp, int8 ev, int8 task_id, int32 step_id, int8 partition_id, int32 node_id, uint64_t frame_id, int64_t input_iter) {
    WRITE_ITEM(double, timestamp);
    WRITE_ITEM(int8, ev);
    WRITE_ITEM(int8, task_id);
    WRITE_ITEM(int32, step_id);
    WRITE_ITEM(int8, partition_id);
    WRITE_ITEM(int32, node_id);
    WRITE_ITEM(uint64_t, frame_id);
    WRITE_ITEM(int64_t, input_iter);
  }

  void write(double timestamp, int8 ev, int8 task_id, int32 step_id, int8 partition_id, int32 node_id) {
    WRITE_ITEM(double, timestamp);
    WRITE_ITEM(int8, ev);
    WRITE_ITEM(int8, task_id);
    WRITE_ITEM(int32, step_id);
    WRITE_ITEM(int8, partition_id);
    WRITE_ITEM(int32, node_id);
  }
};

static std::atomic_int_fast64_t _num_threads(0);
static thread_local int64_t _tid = -1;
static thread_local std::ofstream _meta_fp;
// static thread_local std::ofstream _trace_fp;
// static thread_local std::ostringstream _trace_fp;

// writer for current thread
static thread_local Serializer* _trace_writer;
static Serializer _trace_writers[MAX_NUM_THREADS];

static double currentTimeMillisecond() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return 1e6 * tv.tv_sec + 1.0 * tv.tv_usec;
}

#define BUFFER_SIZE (64 * 1024 * 1024)
//char buffer[BUFFER_SIZE];

void TracingContext::init_thread_if_necessary() {
  if (_tid == -1) {
    _tid = _num_threads.fetch_add(1);

    char buf[256];

    /*
    sprintf(buf, "%s.trace.%ld", _trace_path.c_str(), _tid);
    _trace_fp.open(buf, std::ios::out);
    _trace_fp.rdbuf()->pubsetbuf(new char[BUFFER_SIZE], BUFFER_SIZE);
    _trace_fp.precision(9);
    if(!_trace_fp) {
      LOG(ERROR) << "Failed to open " << buf;
    }
    */
    sprintf(buf, "%s.trace.%ld", _trace_path.c_str(), _tid);
    _trace_writer = _trace_writers + _tid;
    _trace_writer->open(buf, 256*1024*1024);

    std::ofstream* _meta_fp = NULL;
    sprintf(buf, "%s.meta.%ld", _trace_path.c_str(), _tid);
    _meta_fp = new std::ofstream(buf, std::ios::out);
    if(!_meta_fp) {
      LOG(ERROR) << "Failed to open " << buf;
    }
    _trace_writer->_fp = _meta_fp;
  }
}


bool TracingContext::Enabled() {
  return _enabled;
}

void TracingContext::RecordSchedulerBegin(int64_t task_id, int64_t step_id,
    int64_t partition_id, int node_id, uint64_t frame_id, int64_t input_iter) {
  if (! _enabled)
    return;
  init_thread_if_necessary();

  double timestamp = currentTimeMillisecond() - _tracing_context._program_start_time;

  /*
  if (frame_id != 0 || input_iter != 0) {
    _trace_fp << timestamp;
    _trace_fp << SEP << TraceEventType::TRACE_SCHEDULER_BEGIN_WITH_ITER;
    _trace_fp << SEP << task_id;
    _trace_fp << SEP << step_id;
    _trace_fp << SEP << partition_id;
    _trace_fp << SEP << node_id;
    _trace_fp << SEP << frame_id;
    _trace_fp << SEP << input_iter;
    _trace_fp << "\n";
  } else {
    _trace_fp << timestamp;
    _trace_fp << SEP << TraceEventType::TRACE_SCHEDULER_BEGIN;
    _trace_fp << SEP << task_id;
    _trace_fp << SEP << step_id;
    _trace_fp << SEP << partition_id;
    _trace_fp << SEP << node_id;
    _trace_fp << "\n";
  }
  */
  if (frame_id != 0 || input_iter != 0) {
    _trace_writer->write(timestamp, TraceEventType::TRACE_SCHEDULER_BEGIN_WITH_ITER, task_id, step_id, partition_id, node_id, frame_id, input_iter);
  } else {
    _trace_writer->write(timestamp, TraceEventType::TRACE_SCHEDULER_BEGIN, task_id, step_id, partition_id, node_id);
  }
}

void TracingContext::RecordSchedulerEnd(int64_t task_id, int64_t step_id,
    int64_t partition_id, int node_id, uint64_t frame_id, int64_t input_iter) {
  if (! _enabled)
    return;
  init_thread_if_necessary();

  double timestamp = currentTimeMillisecond() - _tracing_context._program_start_time;

//  if (frame_id != 0 || input_iter != 0) {
//    _trace_fp << timestamp;
//    _trace_fp << SEP << TraceEventType::TRACE_SCHEDULER_END_WITH_ITER;
//    _trace_fp << SEP << task_id;
//    _trace_fp << SEP << step_id;
//    _trace_fp << SEP << partition_id;
//    _trace_fp << SEP << node_id;
//    _trace_fp << SEP << frame_id;
//    _trace_fp << SEP << input_iter;
//    _trace_fp << "\n";
//  } else {
//    _trace_fp << timestamp;
//    _trace_fp << SEP << TraceEventType::TRACE_SCHEDULER_END;
//    _trace_fp << SEP << task_id;
//    _trace_fp << SEP << step_id;
//    _trace_fp << SEP << partition_id;
//    _trace_fp << SEP << node_id;
//    _trace_fp << "\n";
//  }
  if (frame_id != 0 || input_iter != 0) {
    _trace_writer->write(timestamp, TraceEventType::TRACE_SCHEDULER_END_WITH_ITER, task_id, step_id, partition_id, node_id, frame_id, input_iter);
  } else {
    _trace_writer->write(timestamp, TraceEventType::TRACE_SCHEDULER_END, task_id, step_id, partition_id, node_id);
  }
}

void TracingContext::RecordComputeBegin(int64_t task_id, int64_t step_id,
    int64_t partition_id, int node_id, uint64_t frame_id, int64_t input_iter,
    bool async) {
  if (! _enabled)
    return;
  init_thread_if_necessary();

  double timestamp = currentTimeMillisecond() - _tracing_context._program_start_time;

//  if (frame_id != 0 || input_iter != 0) {
//    _trace_fp << timestamp;
//    _trace_fp << SEP << TraceEventType::TRACE_COMPUTE_BEGIN_WITH_ITER;
//    _trace_fp << SEP << task_id;
//    _trace_fp << SEP << step_id;
//    _trace_fp << SEP << partition_id;
//    _trace_fp << SEP << node_id;
//    _trace_fp << SEP << frame_id;
//    _trace_fp << SEP << input_iter;
//    _trace_fp << "\n";
//  } else {
//    _trace_fp << timestamp;
//    _trace_fp << SEP << TraceEventType::TRACE_COMPUTE_BEGIN;
//    _trace_fp << SEP << task_id;
//    _trace_fp << SEP << step_id;
//    _trace_fp << SEP << partition_id;
//    _trace_fp << SEP << node_id;
//    _trace_fp << "\n";
//  }
  if (frame_id != 0 || input_iter != 0) {
    _trace_writer->write(timestamp, TraceEventType::TRACE_COMPUTE_BEGIN_WITH_ITER, task_id, step_id, partition_id, node_id, frame_id, input_iter);
  } else {
    _trace_writer->write(timestamp, TraceEventType::TRACE_COMPUTE_BEGIN, task_id, step_id, partition_id, node_id);
  }
}

void TracingContext::RecordComputeEnd(int64_t task_id, int64_t step_id,
    int64_t partition_id, int node_id, uint64_t frame_id, int64_t input_iter,
    bool async) {
  if (! _enabled)
    return;
  init_thread_if_necessary();

  double timestamp = currentTimeMillisecond() - _tracing_context._program_start_time;

//  if (frame_id != 0 || input_iter != 0) {
//    _trace_fp << timestamp;
//    _trace_fp << SEP << TraceEventType::TRACE_COMPUTE_END_WITH_ITER;
//    _trace_fp << SEP << task_id;
//    _trace_fp << SEP << step_id;
//    _trace_fp << SEP << partition_id;
//    _trace_fp << SEP << node_id;
//    _trace_fp << SEP << frame_id;
//    _trace_fp << SEP << input_iter;
//    _trace_fp << "\n";
//  } else {
//    _trace_fp << timestamp;
//    _trace_fp << SEP << TraceEventType::TRACE_COMPUTE_END;
//    _trace_fp << SEP << task_id;
//    _trace_fp << SEP << step_id;
//    _trace_fp << SEP << partition_id;
//    _trace_fp << SEP << node_id;
//    _trace_fp << "\n";
//  }
  if (frame_id != 0 || input_iter != 0) {
    _trace_writer->write(timestamp, TraceEventType::TRACE_COMPUTE_END_WITH_ITER, task_id, step_id, partition_id, node_id, frame_id, input_iter);
  } else {
    _trace_writer->write(timestamp, TraceEventType::TRACE_COMPUTE_END, task_id, step_id, partition_id, node_id);
  }
}

int64_t TracingContext::nextTaskId() {
  return _num_tasks.fetch_add(1);
}

std::ostream& TracingContext::MetaStream() {
  init_thread_if_necessary();
  return *(_trace_writer->_fp);
}

TracingContext::TracingContext() : _num_tasks(0) {
  char* str;

  // Set TF_TRACE_PATH to enable ybw tracing
  str = std::getenv("TF_TRACE_PATH");
  if(str == NULL) {
    LOG(INFO) << "TF_TRACE_PATH not set, default NULL";
  } else {
    _trace_path = string(str);
    _enabled = true;
    LOG(INFO) << "TF_TRACE_PATH=" << _trace_path;
  }
  _program_start_time = currentTimeMillisecond();
  if(sizeof(Serializer) % 64 != 0) {
    LOG(ERROR) << "ERROR: sizeof(Serializer) = " << sizeof(Serializer) << " which not a multiple of 64";
  }
}


TracingContext::~TracingContext() {
  // TODO Auto-generated destructor stub
}

TracingContext _tracing_context;

} /* namespace internal */
} /* namespace tensorflow */


