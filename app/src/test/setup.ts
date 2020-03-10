import * as child from 'child_process'

module.exports = async () => {
  // make sure this matches the redis port in test_config.yml
  const redisProc = child.spawn('redis-server', ['--appendonly', 'no',
    '--save', '', '--port', '6378', '--bind', '127.0.0.1',
    '--protected-mode', 'yes'])
  redisProc.stdout.on('data', (data) => {
    process.stdout.write(data)
  })

  redisProc.stderr.on('data', (data) => {
    process.stdout.write(data)
  })
}
