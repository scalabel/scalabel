import { Sensor } from "../../src/common/sensor"
import { SensorType } from "../../src/types/state"
import * as THREE from "three"
import {
  extrinsicsFromExport,
  intrinsicsFromExport
} from "../../src/server/bdd_type_transformers"

describe("test sensor creation and transformations", () => {
  test("create sensor with no extrinsics", () => {
    const sensorType: SensorType = {
      id: 0,
      name: "sensor0",
      type: "image"
    }
    const sensor = Sensor.fromSensorType(sensorType)
    expect(sensor.hasExtrinsics()).toBeFalsy()
    const point = new THREE.Vector3(1, 2, 3)
    const transformed = sensor.transform(point)
    expect(point.toArray()).toEqual(transformed.toArray())
  })
  test("create sensor with extrinsics", () => {
    const sensorType: SensorType = {
      id: 0,
      name: "sensor0",
      type: "image",
      intrinsics: intrinsicsFromExport({
        focal: [721.5377197265625, 721.5377197265625],
        center: [609.559326171875, 172.85400390625]
      }),
      extrinsics: extrinsicsFromExport({
        location: [
          -0.0027968167666705956, -0.0751087909738869, -0.27213280776889376
        ],
        rotation: [0.011902515224334161, -1.5603441219160237, 1.548328901848571]
      })
    }
    const sensor = Sensor.fromSensorType(sensorType)
    expect(sensor.hasExtrinsics()).toBeTruthy()
    const point = new THREE.Vector3(1, 2, 3)
    const expectedPoint = new THREE.Vector3(
      -2.0341408195370954,
      -3.0431974451317565,
      0.7594151957035378
    )
    const transformed = sensor.transform(point)
    const inverseTransform = sensor.inverseTransform(transformed)
    expect(expectedPoint.toArray()).toEqual(transformed.toArray())
    const eps = 0.00001
    expect(point.toArray()[0] - inverseTransform.toArray()[0]).toBeLessThan(eps)
    expect(point.toArray()[1] - inverseTransform.toArray()[1]).toBeLessThan(eps)
    expect(point.toArray()[2] - inverseTransform.toArray()[2]).toBeLessThan(eps)
  })
})
