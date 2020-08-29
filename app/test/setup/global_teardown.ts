import * as child from "child_process"

module.exports = async () => {
  child.spawn("pkill", ["redis-server"])
}
