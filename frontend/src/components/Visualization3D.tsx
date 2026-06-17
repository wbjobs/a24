import React, { useRef, useMemo, useEffect, useState } from 'react'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { OrbitControls, Text } from '@react-three/drei'
import * as THREE from 'three'

interface Visualization3DProps {
  gridData: {
    x: number[]
    t: number[]
    u: number[][]
  } | null
  isAnimating: boolean
  currentTimeIndex: number
  colorScheme: string
}

function SurfaceMesh({
  gridData,
  currentTimeIndex,
  colorScheme,
}: {
  gridData: { x: number[]; t: number[]; u: number[][] }
  currentTimeIndex: number
  colorScheme: string
}) {
  const meshRef = useRef<THREE.Mesh>(null)

  const { geometry, uMin, uMax } = useMemo(() => {
    const xArr = gridData.x
    const tArr = gridData.t
    const uArr = gridData.u
    const nx = xArr.length
    const nt = tArr.length
    const xMin = Math.min(...xArr)
    const xMax = Math.max(...xArr)
    const tMin = Math.min(...tArr)
    const tMax = Math.max(...tArr)
    const uValues = uArr.flat()
    const uMinVal = Math.min(...uValues)
    const uMaxVal = Math.max(...uValues)

    const vertices: number[] = []
    const colors: number[] = []
    const indices: number[] = []

    const xRange = xMax - xMin || 1
    const tRange = tMax - tMin || 1
    const uRange = uMaxVal - uMinVal || 1

    const clampT = Math.min(currentTimeIndex, nt - 1)

    for (let j = 0; j <= clampT; j++) {
      for (let i = 0; i < nx; i++) {
        const xn = (xArr[i] - xMin) / xRange * 4 - 2
        const tn = (tArr[j] - tMin) / tRange * 4 - 2
        const un = (uArr[j][i] - uMinVal) / uRange * 2 - 1
        vertices.push(xn, un, tn)

        const colorVal = (uArr[j][i] - uMinVal) / uRange
        const [r, g, b] = getColor(colorVal, colorScheme)
        colors.push(r, g, b)
      }
    }

    for (let j = 0; j < clampT; j++) {
      for (let i = 0; i < nx - 1; i++) {
        const a = j * nx + i
        const b = j * nx + i + 1
        const c = (j + 1) * nx + i
        const d = (j + 1) * nx + i + 1
        indices.push(a, c, b)
        indices.push(b, c, d)
      }
    }

    const geo = new THREE.BufferGeometry()
    geo.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3))
    geo.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3))
    geo.setIndex(indices)
    geo.computeVertexNormals()

    return { geometry: geo, uMin: uMinVal, uMax: uMaxVal }
  }, [gridData, currentTimeIndex, colorScheme])

  useEffect(() => {
    return () => {
      geometry.dispose()
    }
  }, [geometry])

  return (
    <mesh ref={meshRef} geometry={geometry}>
      <meshStandardMaterial
        vertexColors
        side={THREE.DoubleSide}
        metalness={0.1}
        roughness={0.6}
      />
    </mesh>
  )
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

function AxisLabels({ gridData }: { gridData: { x: number[]; t: number[]; u: number[][] } }) {
  return (
    <group>
      <Text position={[-2.5, -1.5, 0]} fontSize={0.2} color="#a0a0b8">
        x
      </Text>
      <Text position={[0, -1.8, 2.5]} fontSize={0.2} color="#a0a0b8">
        t
      </Text>
      <Text position={[-2.5, 0, -0.5]} fontSize={0.2} color="#a0a0b8">
        u(x,t)
      </Text>
    </group>
  )
}

function GridLines({ gridData }: { gridData: { x: number[]; t: number[]; u: number[][] } }) {
  const lineGeo = useMemo(() => {
    const points: THREE.Vector3[] = []
    const xRange = Math.max(...gridData.x) - Math.min(...gridData.x) || 1
    const xMin = Math.min(...gridData.x)
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
    const geo = new THREE.BufferGeometry().setFromPoints(points)
    return geo
  }, [gridData])

  return (
    <lineSegments geometry={lineGeo}>
      <lineBasicMaterial color="#2d2d4a" transparent opacity={0.4} />
    </lineSegments>
  )
}

const Visualization3D: React.FC<Visualization3DProps> = ({
  gridData,
  isAnimating,
  currentTimeIndex,
  colorScheme,
}) => {
  if (!gridData) {
    return (
      <div className="viz-container">
        <div className="empty-state">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
            <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z" />
            <polyline points="3.27 6.96 12 12.01 20.73 6.96" />
            <line x1="12" y1="22.08" x2="12" y2="12" />
          </svg>
          <p>运行求解器以查看3D可视化</p>
        </div>
      </div>
    )
  }

  return (
    <div className="viz-container">
      <Canvas
        camera={{ position: [4, 3, 4], fov: 50 }}
        gl={{ antialias: true, alpha: true }}
        style={{ background: 'var(--bg-primary)' }}
      >
        <ambientLight intensity={0.4} />
        <directionalLight position={[5, 5, 5]} intensity={0.8} />
        <directionalLight position={[-3, 3, -3]} intensity={0.3} />
        <SurfaceMesh
          gridData={gridData}
          currentTimeIndex={currentTimeIndex}
          colorScheme={colorScheme}
        />
        <AxisLabels gridData={gridData} />
        <GridLines gridData={gridData} />
        <OrbitControls
          enableDamping
          dampingFactor={0.1}
          rotateSpeed={0.5}
          zoomSpeed={0.8}
        />
      </Canvas>
    </div>
  )
}

export default Visualization3D
