import React, { useRef, useMemo, useEffect } from 'react'
import { Canvas } from '@react-three/fiber'
import { OrbitControls, Text } from '@react-three/drei'
import * as THREE from 'three'

export type VizMode = 'mean' | 'uncertainty' | 'std'

interface Visualization3DProps {
  gridData: {
    x: number[]
    t: number[]
    u: number[][]
    u_std?: number[][]
    u_uncertainty?: number[][]
    has_uncertainty?: boolean
  } | null
  isAnimating: boolean
  currentTimeIndex: number
  colorScheme: string
  vizMode?: VizMode
  showUncertaintyBand?: boolean
}

function getColor(value: number, scheme: string): [number, number, number] {
  const v = Math.max(0, Math.min(1, value))
  switch (scheme) {
    case 'coolwarm':
      if (v < 0.5) {
        const t = v * 2
        return [0.2 + t * 0.3, 0.3 + t * 0.3, 0.8 + t * 0.2]
      } else {
        const t = (v - 0.5) * 2
        return [0.8 + t * 0.2, 0.4 - t * 0.3, 0.6 - t * 0.5]
      }
    case 'viridis':
      if (v < 0.25) {
        const t = v * 4
        return [0.267 + t * 0.004, 0.004 + t * 0.396, 0.329 + t * 0.324]
      } else if (v < 0.5) {
        const t = (v - 0.25) * 4
        return [0.271 - t * 0.155, 0.400 + t * 0.288, 0.653 - t * 0.191]
      } else if (v < 0.75) {
        const t = (v - 0.5) * 4
        return [0.116 + t * 0.511, 0.688 + t * 0.149, 0.462 - t * 0.276]
      } else {
        const t = (v - 0.75) * 4
        return [0.627 + t * 0.327, 0.837 - t * 0.087, 0.186 - t * 0.144]
      }
    case 'uncertainty':
      return [
        Math.min(1, v * 2),
        Math.min(1, 1.5 - Math.abs(v - 0.5) * 3),
        Math.max(0, 1 - v * 2),
      ]
    case 'plasma':
    default:
      if (v < 0.33) {
        const t = v * 3
        return [0.050 + t * 0.483, 0.030 + t * 0.047, 0.528 + t * 0.206]
      } else if (v < 0.66) {
        const t = (v - 0.33) * 3
        return [0.533 + t * 0.422, 0.077 + t * 0.576, 0.734 - t * 0.354]
      } else {
        const t = (v - 0.66) * 3
        return [0.955 - t * 0.132, 0.653 + t * 0.277, 0.380 - t * 0.295]
      }
  }
}

interface MeshInput {
  gridData: any
  currentTimeIndex: number
  colorScheme: string
  vizMode: VizMode
  showBand: boolean
}

function SurfaceMesh({ gridData, currentTimeIndex, colorScheme, vizMode, showBand }: MeshInput) {
  const meshRef = useRef<THREE.Mesh>(null)
  const bandMeshRef = useRef<THREE.Mesh>(null)
  const bandTopRef = useRef<THREE.Mesh>(null)
  const bandBottomRef = useRef<THREE.Mesh>(null)

  const { mainGeom, bandGeom, bandTopGeom, bandBottomGeom, colorMin, colorMax } = useMemo(() => {
    const xArr: number[] = gridData.x
    const tArr: number[] = gridData.t
    const uArr: number[][] = gridData.u
    const uncArr: number[][] = gridData.u_uncertainty || gridData.u_std || []
    const nx = xArr.length
    const nt = tArr.length
    const xMin = Math.min(...xArr)
    const xMax = Math.max(...xArr)
    const tMin = Math.min(...tArr)
    const tMax = Math.max(...tArr)

    let values: number[][]
    let scheme = colorScheme

    if (vizMode === 'mean') {
      values = uArr
    } else if (vizMode === 'uncertainty' || vizMode === 'std') {
      if (uncArr.length > 0) {
        values = uncArr
        scheme = 'uncertainty'
      } else {
        values = uArr
      }
    } else {
      values = uArr
    }

    const flatValues = values.flat()
    const minVal = Math.min(...flatValues)
    const maxVal = Math.max(...flatValues)
    const valueRange = maxVal - minVal || 1

    const xRange = xMax - xMin || 1
    const tRange = tMax - tMin || 1
    const uRange = (Math.max(...uArr.flat()) - Math.min(...uArr.flat())) || 1
    const uMinVal = Math.min(...uArr.flat())

    const clampT = Math.min(currentTimeIndex, nt - 1)

    const mainVerts: number[] = []
    const mainColors: number[] = []
    const mainIdx: number[] = []

    const bandVerts: number[] = []
    const bandColors: number[] = []
    const bandIdx: number[] = []

    const bandTopVerts: number[] = []
    const bandTopColors: number[] = []
    const bandTopIdx: number[] = []

    const bandBottomVerts: number[] = []
    const bandBottomColors: number[] = []
    const bandBottomIdx: number[] = []

    const hasUnc = uncArr.length === nt && uncArr[0].length === nx && showBand

    for (let j = 0; j <= clampT; j++) {
      for (let i = 0; i < nx; i++) {
        const xn = (xArr[i] - xMin) / xRange * 4 - 2
        const tn = (tArr[j] - tMin) / tRange * 4 - 2
        const uBase = (uArr[j][i] - uMinVal) / uRange * 2 - 1

        const value = values[j][i]
        const colorVal = (value - minVal) / valueRange
        const [r, g, b] = getColor(colorVal, scheme)

        mainVerts.push(xn, uBase, tn)
        mainColors.push(r, g, b)

        if (hasUnc) {
          const uncVal = (uncArr[j][i] || 0) / uRange * 2
          const uncColor: [number, number, number] = [0.9, 0.3, 0.3]

          bandTopVerts.push(xn, uBase + uncVal, tn)
          bandTopColors.push(uncColor[0], uncColor[1], uncColor[2])

          bandBottomVerts.push(xn, uBase - uncVal, tn)
          bandBottomColors.push(uncColor[0], uncColor[1], uncColor[2])

          const baseIdx = j * nx + i
          bandVerts.push(xn, uBase, tn)
          bandColors.push(uncColor[0] * 0.5, uncColor[1] * 0.5, uncColor[2] * 0.5)
        }
      }
    }

    for (let j = 0; j < clampT; j++) {
      for (let i = 0; i < nx - 1; i++) {
        const a = j * nx + i
        const b = j * nx + i + 1
        const c = (j + 1) * nx + i
        const d = (j + 1) * nx + i + 1
        mainIdx.push(a, c, b)
        mainIdx.push(b, c, d)
        if (hasUnc) {
          bandTopIdx.push(a, c, b)
          bandTopIdx.push(b, c, d)
          bandBottomIdx.push(a, c, b)
          bandBottomIdx.push(b, c, d)
          bandIdx.push(a, c, b)
          bandIdx.push(b, c, d)
        }
      }
    }

    const makeGeo = (v: number[], c: number[], idx: number[]) => {
      const g = new THREE.BufferGeometry()
      g.setAttribute('position', new THREE.Float32BufferAttribute(v, 3))
      g.setAttribute('color', new THREE.Float32BufferAttribute(c, 3))
      g.setIndex(idx)
      g.computeVertexNormals()
      return g
    }

    return {
      mainGeom: makeGeo(mainVerts, mainColors, mainIdx),
      bandGeom: hasUnc ? makeGeo(bandVerts, bandColors, bandIdx) : null,
      bandTopGeom: hasUnc ? makeGeo(bandTopVerts, bandTopColors, bandTopIdx) : null,
      bandBottomGeom: hasUnc ? makeGeo(bandBottomVerts, bandBottomColors, bandBottomIdx) : null,
      colorMin: minVal,
      colorMax: maxVal,
    }
  }, [gridData, currentTimeIndex, colorScheme, vizMode, showBand])

  useEffect(() => {
    return () => {
      mainGeom.dispose()
      bandGeom?.dispose()
      bandTopGeom?.dispose()
      bandBottomGeom?.dispose()
    }
  }, [mainGeom, bandGeom, bandTopGeom, bandBottomGeom])

  return (
    <group>
      <mesh ref={meshRef} geometry={mainGeom}>
        <meshStandardMaterial
          vertexColors
          side={THREE.DoubleSide}
          metalness={0.08}
          roughness={0.7}
          flatShading={false}
        />
      </mesh>
      {bandTopGeom && (
        <mesh ref={bandTopRef} geometry={bandTopGeom}>
          <meshBasicMaterial
            vertexColors
            side={THREE.DoubleSide}
            transparent
            opacity={0.25}
          />
        </mesh>
      )}
      {bandBottomGeom && (
        <mesh ref={bandBottomRef} geometry={bandBottomGeom}>
          <meshBasicMaterial
            vertexColors
            side={THREE.DoubleSide}
            transparent
            opacity={0.25}
          />
        </mesh>
      )}
    </group>
  )
}

function AxisLabels() {
  return (
    <group>
      <Text position={[-2.6, -1.5, 0]} fontSize={0.2} color="#a0a0b8" anchorX="center">
        x
      </Text>
      <Text position={[0, -1.9, 2.6]} fontSize={0.2} color="#a0a0b8" anchorX="center">
        t
      </Text>
      <Text position={[-2.7, 0.05, -0.3]} fontSize={0.2} color="#a0a0b8" anchorX="center">
        u(x,t)
      </Text>
    </group>
  )
}

function GridLines() {
  const lineGeo = useMemo(() => {
    const points: THREE.Vector3[] = []
    for (let i = 0; i <= 10; i++) {
      const xn = (i / 10) * 4 - 2
      points.push(new THREE.Vector3(xn, -1.5, -2))
      points.push(new THREE.Vector3(xn, -1.5, 2))
    }
    for (let i = 0; i <= 10; i++) {
      const zn = (i / 10) * 4 - 2
      points.push(new THREE.Vector3(-2, -1.5, zn))
      points.push(new THREE.Vector3(2, -1.5, zn))
    }
    return new THREE.BufferGeometry().setFromPoints(points)
  }, [])

  return (
    <lineSegments geometry={lineGeo}>
      <lineBasicMaterial color="#2d2d4a" transparent opacity={0.5} />
    </lineSegments>
  )
}

const Visualization3D: React.FC<Visualization3DProps> = ({
  gridData,
  currentTimeIndex,
  colorScheme,
  vizMode = 'mean',
  showUncertaintyBand = true,
}) => {
  if (!gridData) {
    return (
      <div className="viz-container">
        <div className="empty-state">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" style={{ opacity: 0.4 }}>
            <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z" />
            <polyline points="3.27 6.96 12 12.01 20.73 6.96" />
            <line x1="12" y1="22.08" x2="12" y2="12" />
          </svg>
          <p>运行求解器以查看3D可视化</p>
        </div>
      </div>
    )
  }

  const hasUnc = gridData.has_uncertainty && (gridData.u_uncertainty || gridData.u_std)

  return (
    <div className="viz-container">
      <Canvas
        camera={{ position: [4.2, 3.2, 4.2], fov: 48 }}
        gl={{ antialias: true, alpha: true, powerPreference: 'high-performance' }}
        style={{ background: 'var(--bg-primary)' }}
      >
        <ambientLight intensity={0.55} />
        <directionalLight position={[6, 7, 5]} intensity={0.9} />
        <directionalLight position={[-4, 4, -4]} intensity={0.35} />
        <directionalLight position={[0, -3, 0]} intensity={0.15} />

        <SurfaceMesh
          gridData={gridData}
          currentTimeIndex={currentTimeIndex}
          colorScheme={colorScheme}
          vizMode={vizMode}
          showBand={!!(hasUnc && showUncertaintyBand && vizMode === 'mean')}
        />
        <AxisLabels />
        <GridLines />
        <OrbitControls
          enableDamping
          dampingFactor={0.08}
          rotateSpeed={0.55}
          zoomSpeed={0.85}
          panSpeed={0.7}
          enablePan
          minDistance={2}
          maxDistance={15}
        />
      </Canvas>
    </div>
  )
}

export default Visualization3D
