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


enum TraceElementType {OP_BEGIN=1, OP_END};

struct TraceElement {
  TraceElementType _type;
  int _node_id;
  int64_t _step_id;
  double _timestamp;

  TraceElement(TraceElementType type, int node_id, int64_t step_id, double timestamp) :
    _type(type), _node_id(node_id), _step_id(step_id), _timestamp(timestamp) {}
};


static std::ostream& operator<<(std::ostream& out, const TraceElement e) {
  out << e._type << "," << e._node_id << "," << e._step_id << "," << e._timestamp;
  return out;
}


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
   * in_node_in_list  a list of the id of the dependency nodes
   */
  void RecordBegin(int node_id, int64_t step_id, const std::vector<int>& in_node_id_list);
  void RecordEnd(int node_id, int64_t step_id);

  TracingContext();
  virtual ~TracingContext();
};

extern TracingContext _tracing_context;


} /* namespace internal */
} /* namespace tensorflow */

#endif /* TENSORFLOW_CORE_PLATFORM_DEFAULT_TRACINGCONTEXT_H_ */

