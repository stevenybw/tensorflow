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


namespace tensorflow {
namespace internal {

static double currentTimeMillisecond() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return 1e6 * tv.tv_sec + 1.0 * tv.tv_usec;
}

#define BUFFER_SIZE 8192
char buffer[BUFFER_SIZE];

TracingContext::TracingContext() {
  char* str;

  // Set TF_TRACE_PATH to enable ybw tracing
  str = std::getenv("TF_TRACE_PATH");
  if(str == NULL) {
    LOG(INFO) << "TF_TRACE_PATH not set, default NULL";
  } else {
    _trace_path = string(str);
    _enabled = true;
    _fp.open(_trace_path, std::ios::out | std::ios::app);
    _fp.rdbuf()->pubsetbuf(buffer, BUFFER_SIZE);
    _fp.precision(9);
    LOG(INFO) << "TF_TRACE_PATH=" << _trace_path;
  }
  _program_start_time = currentTimeMillisecond();
}

void TracingContext::RecordBegin(int node_id, int64_t step_id, const string& node_name,
    const string& type_string, const string& assigned_device_name, const std::vector<TracingNode>& in_node_id_list) {
  Lock l(_mu);
  if(_enabled) {
    double beginTime = currentTimeMillisecond() - _program_start_time;
    _fp << TraceElementType(OP_BEGIN) << "," << node_id << "," << step_id << "," << node_name << "," << type_string << "," << assigned_device_name << "," << beginTime << ",{ ";
    for (TracingNode in_node_id : in_node_id_list) {
      _fp << in_node_id._node_id << "!" << in_node_id._node_name << "!" << in_node_id._assigned_device_name << " ";
    }
    _fp << "}\n";
  }
}

void TracingContext::RecordSendRecvBegin(int node_id, int64_t step_id,
    const string& node_name, const string& type_string,
    const string& assigned_device_name,
    const std::vector<TracingNode>& in_node_id_list, const string& full_key) {
  Lock l(_mu);
  if(_enabled) {
    double beginTime = currentTimeMillisecond() - _program_start_time;
    _fp << TraceElementType(OP_BEGIN) << "," << node_id << "," << step_id << "," << node_name << "," << type_string << "," << assigned_device_name << "," << beginTime << ",{ ";
    for (TracingNode in_node_id : in_node_id_list) {
      _fp << in_node_id._node_id << "!" << in_node_id._node_name << "!" << in_node_id._assigned_device_name << " ";
    }
    _fp << "}" << "," << full_key << "\n";
  }
}

void TracingContext::RecordEnd(int node_id, int64_t step_id, const string& assigned_device_name) {
  Lock l(_mu);
  if(_enabled) {
    double endTime = currentTimeMillisecond() - _program_start_time;
    _fp << TraceElementType(OP_END) << "," << node_id << "," << step_id << "," << assigned_device_name << "," << endTime << "\n";
  }
}

TracingContext::~TracingContext() {
  // TODO Auto-generated destructor stub
}

TracingContext _tracing_context;

} /* namespace internal */
} /* namespace tensorflow */

