import { ShapeTypeName } from '../../../js/common/types'
import { Box2D, Handles } from '../../../js/drawable/2d/box2d'
import { Label2DList } from '../../../js/drawable/2d/label2d_list'
import { Rect2D } from '../../../js/drawable/2d/rect2d'
import { Size2D } from '../../../js/math/size2d'
import { Vector2D } from '../../../js/math/vector2d'

/** Make box with default parameters */
function makeDefaultBox (
  labelList: Label2DList, start: Vector2D = new Vector2D()
) {
  const box = new Box2D(labelList)
  box.initTemp(0, -1, [], {}, [0, 0, 0], start)
  return box
}

test('Box2D initialization', () => {
  const labelList = new Label2DList()
  const box = makeDefaultBox(labelList)
  box.drag(new Vector2D(10, 10), new Size2D(100, 100))

  const shapes = box.shapes()
  expect(shapes.length).toEqual(1)
  expect(shapes[0].typeName).toEqual(ShapeTypeName.RECT)

  const rect = shapes[0] as Rect2D
  expect(rect.x).toEqual(0)
  expect(rect.y).toEqual(0)
  expect(rect.w).toEqual(10)
  expect(rect.h).toEqual(10)

  expect(labelList.updatedLabels.size).toEqual(1)
  expect(labelList.updatedShapes.size).toEqual(1)
})

test('Box2D corner editing', () => {
  const labelList = new Label2DList()
  const limit = new Size2D(100, 100)
  const box = makeDefaultBox(labelList, new Vector2D(50, 50))
  box.drag(new Vector2D(10, 10), limit)

  const rect = box.shapes()[0] as Rect2D

  box.setHighlighted(true, Handles.TOP_LEFT)
  box.drag(new Vector2D(-5, -5), limit)
  expect(rect.x).toEqual(45)
  expect(rect.y).toEqual(45)
  expect(rect.w).toEqual(15)
  expect(rect.h).toEqual(15)

  box.setHighlighted(true, Handles.TOP_RIGHT)
  box.drag(new Vector2D(5, -5), limit)
  expect(rect.x).toEqual(45)
  expect(rect.y).toEqual(40)
  expect(rect.w).toEqual(20)
  expect(rect.h).toEqual(20)

  box.setHighlighted(true, Handles.BOTTOM_LEFT)
  box.drag(new Vector2D(5, -5), limit)
  expect(rect.x).toEqual(50)
  expect(rect.y).toEqual(40)
  expect(rect.w).toEqual(15)
  expect(rect.h).toEqual(15)

  box.setHighlighted(true, Handles.BOTTOM_RIGHT)
  box.drag(new Vector2D(-5, -5), limit)
  expect(rect.x).toEqual(50)
  expect(rect.y).toEqual(40)
  expect(rect.w).toEqual(10)
  expect(rect.h).toEqual(10)

  box.drag(new Vector2D(-20, -20), limit)
  expect(rect.x).toEqual(40)
  expect(rect.y).toEqual(30)
  expect(rect.w).toEqual(10)
  expect(rect.h).toEqual(10)

  box.drag(new Vector2D(5, 5), limit)
  expect(rect.x).toEqual(45)
  expect(rect.y).toEqual(35)
  expect(rect.w).toEqual(5)
  expect(rect.h).toEqual(5)
})
