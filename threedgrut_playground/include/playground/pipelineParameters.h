// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved. SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <optix.h>

#include <3dgrt/pipelineParameters.h>
#include <3dgrt/tensorAccessor.h>
#include <playground/cutexture.h>
#include <playground/pipelineDefinitions.h>

struct alignas(16) PBRMaterial {
  cudaTextureObject_t diffuseTexture;           // 8 bytes -> offset 8
  cudaTextureObject_t emissiveTexture;          // 8 bytes -> offset 16
  cudaTextureObject_t metallicRoughnessTexture; // 8 bytes -> offset 24
  cudaTextureObject_t normalTexture;            // 8 bytes -> offset 32

  float4 diffuseFactor;     // 16 bytes -> offset 48
  float3 emissiveFactor;    // 12 bytes -> offset 60
  float metallicFactor;     // 4 bytes  -> offset 64
  float roughnessFactor;    // 4 bytes  -> offset 68
  float transmissionFactor; // 4 bytes  -> offset 72
  float ior;                // 4 bytes  -> offset 76
  float alphaCutoff;        // 4 bytes  -> offset 80
  unsigned int alphaMode; // see GltfAlphaMode         // 4 bytes  -> offset 84

  bool useDiffuseTexture;           // 1 byte -> offset 85
  bool useEmissiveTexture;          // 1 byte -> offset 86
  bool useMetallicRoughnessTexture; // 1 byte -> offset 87
  bool useNormalTexture;            // 1 byte -> offset 88

  float2 pad0; // 8 bytes -> offset 96
               // --> mem 16 byte aligned
};

// -- Light sources (Phase II: directional) --
// Explicit byte layout locks host(std::vector)/device ABI parity and keeps the
// offset map readable. Note: CUDA float3 is 4-byte aligned (only float4 is 16B),
// so there is no implicit padding here; pad0 just places direction at offset 16.
enum PlaygroundLightType {
  PGRNDLightNone = 0,
  PGRNDLightDirectional = 1
  // future: PGRNDLightPoint = 2, PGRNDLightSpot = 3, PGRNDLightEnvmapIS = 4
};

struct alignas(16) PlaygroundLight {
  unsigned int type;          // 4 bytes  -> offset 4  ; PlaygroundLightType
  float        pad0[3];       // 12 bytes -> offset 16 ; pad so direction starts at offset 16
  float3       direction;     // 12 bytes -> offset 28 ; unit vec, shading point -> light (world)
  float        intensity;     // 4 bytes  -> offset 32 ; scalar multiplier
  float3       color;         // 12 bytes -> offset 44 ; linear RGB
  float        angularRadius; // 4 bytes  -> offset 48 ; radians; 0=hard, >0=soft (Phase 4)
};                            // sizeof == 48, 16-byte aligned
static_assert(sizeof(PlaygroundLight) == 48,
              "PlaygroundLight must be 48 bytes for host/device ABI parity");

struct PlaygroundPipelineParameters : PipelineParameters {
  PackedTensorAccessor32<float, 3> rayMaxT; ///< ray max t for termination
  cudaTextureObject_t envmap; // for envmaps and solid background color
  float2 envmapOffset;        // rotates env map along (theta, phi) axis

  // -- Playground specific launch params --
  OptixTraversableHandle
      triHandle; // Handle to BVH of mesh primitives: mirrors, glasses, pbr..

  unsigned int playgroundOpts; // see PlaygroundRenderOptions
  unsigned int maxPBRBounces;  // Maximum PBR ray iterations (reflections,
                               // transmissions & refractions)
  float shadowMin;             // shadow-catcher floor: radiance *= shadowMin +
                               // (1-shadowMin)*visibility (0 = shadows reach black)
  unsigned int shadowSpp;      // soft-shadow occlusion samples per light
  PackedTensorAccessor32<int32_t, 4>
      trace_state; // Scratch buffer, stores current render pass per ray
  PackedTensorAccessor32<int32_t, 2>
      triangles; // Primitive index -> vertex indices

  // Per vertex attributes
  PackedTensorAccessor32<float, 2> vNormals; // vertex normals
  PackedTensorAccessor32<bool, 2>
      vHasTangents; // has precomputed vertex tangents;
  PackedTensorAccessor32<float, 2> vTangents; // vertex tangents

  // Materials
  PackedTensorAccessor32<float, 3>
      matUV; // [F,3,2]: triangular face X vertex X 2d uv
  PackedTensorAccessor32<int32_t, 2> matID; // id of material to use, per vertex
  PBRMaterial *materials;                   // dynamic array of materials
  unsigned int numMaterials;

  // Per triangle attributes
  PackedTensorAccessor32<int32_t, 2> primType; // see PlaygroundPrimitiveTypes
  PackedTensorAccessor32<float, 2>
      refractiveIndex; // glass refraction, higher -> thicker glass

  // -- Light sources (Phase II) --
  const PlaygroundLight *lights; // device array; nullptr when numLights == 0
  unsigned int numLights;        // number of lights, may be 0
};
