/*
 * tracing_context.h
 *
 *  Created on: 2016年11月22日
 *      Author: Bowen Yu <yubowen15@foxmail.com>
 */

#ifndef TENSORFLOW_CORE_PLATFORM_DEFAULT_TRACINGCONTEXT_H_
#define TENSORFLOW_CORE_PLATFORM_DEFAULT_TRACINGCONTEXT_H_

#include <iostream>
#include <fstream>
#include <string>
#include <mutex>
#include <vector>

using std::string;

namespace tensorflow {
namespace internal {


enum TraceElementType {OP_BEGIN=1, OP_END, OP_SENDRECV_BEGIN, OP_SENDRECV_END, OP_NEW_EXECUTORS_AND_KEYS};

/*
 * TracingNode contains all the information sufficient to identify a specific node.
 */
struct TracingNode {
  int _node_id;
  string _node_name;
  string _assigned_device_name;
  TracingNode(int node_id, string node_name, string assigned_device_name) :
    _node_id(node_id), _node_name(node_name), _assigned_device_name(assigned_device_name) {}
};

class TracingContext {
private:
  typedef std::lock_guard<std::mutex>  Lock;

  std::mutex _mu;
  string _trace_path;
  std::ofstream _fp;
  bool _enabled = false;

public:
  double _program_start_time;

  /*
   * params:
   *
   * node_id  id of the current node being traced
   * step_id  current step id
   * node_name    name of the node being traced
   * type_string  type of the node being traced
   * assigned_device_name    device name that the node is assigned to
   * in_node_in_list  a list of the id of the dependency nodes
   */
  void RecordBegin(int node_id, int64_t step_id, const string& node_name, const string& type_string, const string& assigned_device_name, const std::vector<TracingNode>& in_node_id_list, uintptr_t run_id);
  void RecordEnd(int node_id, int64_t step_id, const string& assigned_device_name);

  /*
   * An extra param, full_key is concerned compared to record for SendRecv operation.
   * This information helps us connect _Send to corresponding _Recv when analyzing.
   */
  void RecordSendRecvBegin(int node_id, int64_t step_id, const string& node_name, const string& type_string, const string& assigned_device_name, const std::vector<TracingNode>& in_node_id_list, const string& full_key, uintptr_t run_id);
  void RecordSendRecvEnd(int node_id, int64_t step_id, const string& assigned_device_name);

  /*
   * Record the run_id and its corresponding target_nodes names.
   */
  void RecordNewExecutorsAndKeys(uintptr_t run_id, const std::vector<string>& target_nodes);

  TracingContext();
  virtual ~TracingContext();
};

extern TracingContext _tracing_context;


} /* namespace internal */
} /* namespace tensorflow */

#endif /* TENSORFLOW_CORE_PLATFORM_DEFAULT_TRACINGCONTEXT_H_ */

