#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <random>
#include <vector>
#include <algorithm>
#include <ctime>
#include <iostream>
#include <cmath>
#include <fstream>

#include "kernels.cuh"

#define numPatterns 100
#define numDriven 50
#define valDriven 100

#define maxExcitWeight 300
#define maxInhibWeight -400

#define offset 3

#define tauPSPF 15.0
#define PSPFCutoff 10
#define writeRawData false



int main()
{
	int numNeurons = 1000;
	int numExcit = 800;
	int T = 2000;
	int equilizationTime = 100;
	int transientTime = 300;
	int maxDelay = 15;

	/* CUDA Parameters */
	int numThreads = 512;

	/* Neurons */
	float *h_v, *d_v, *h_u, *d_u, *h_I, *d_I, *h_driven, *d_driven;
	bool *d_cf, *h_cf;

	h_v = new float[numNeurons];
	h_u = new float[numNeurons];
	h_I = new float[numNeurons*maxDelay];
	h_cf = new bool[numNeurons];
	h_driven = new float[numNeurons];

	bool **SpikeTrainYard = new bool*[T];
	float **VoltageTrace = new float *[T];
	for (int i = 0; i < numNeurons; i++)
	{
		for (int j = 0; j < maxDelay; j++)
		{
			h_I[i*maxDelay + j] = 0;
		}
		h_v[i] = -60;
		h_u[i] = 0;
		h_cf[i] = false;
		if (i < 100)
		{
			h_driven[i] = valDriven;
		}
		else
		{
			h_driven[i] = 0;
		}
	}

	for (int t = 0; t < T; t++)
	{
		SpikeTrainYard[t] = new bool[numNeurons];
		VoltageTrace[t] = new float[numNeurons];
	}


	/* Edges */

	std::vector<int> h_source; int *d_source;
	std::vector<int> h_target; int *d_target;
	std::vector<float> h_weight; float *d_weight;
	std::vector<int> h_delay; int *d_delay;

	std::mt19937 rd(time(NULL));
	std::uniform_real_distribution<float> dist(0.0, 1.0);
	std::uniform_int_distribution<int> intDist(1, maxDelay);

	for (int n = 0; n < numNeurons; n++)
	{
		for (int m = 0; m < numNeurons; m++)
		{
			if (n != m)
			{

				if (dist(rd) < .2)
				{
					h_source.push_back(n);
					h_target.push_back(m);
					h_delay.push_back(intDist(rd));
					if (n < numExcit)
					{
						h_weight.push_back(dist(rd) * maxExcitWeight);
					}
					else
					{
						h_weight.push_back(dist(rd) * maxInhibWeight);
					}
				}

			}
		}
	}

	int numEdges = h_source.size();

	/* Pseudometric Code */

	float *d_PSPF;
	float ***PSPFs = new float**[numPatterns];
	for (int i = 0; i < numPatterns; i++)
	{
		PSPFs[i] = new float*[T];
		for (int t = 0; t < T; t++)
		{
			PSPFs[i][t] = new float[numNeurons];
		}
	}

	float normalizer = std::max(abs(maxExcitWeight), abs(maxInhibWeight));

	/* Population VanRossum */

	float *d_PVR;
	float ***PVRs = new float**[numPatterns];
	for (int i = 0; i < numPatterns; i++)
	{
		PVRs[i] = new float*[T];
		for (int t = 0; t < T; t++)
		{
			PVRs[i][t] = new float[numNeurons];
		}
	}



	/* CUDA Memory Functions */

	cudaMalloc((void**)&d_v, numNeurons * sizeof(float));
	cudaMalloc((void**)&d_u, numNeurons * sizeof(float));
	cudaMalloc((void**)&d_I, numNeurons * maxDelay * sizeof(float));
	cudaMalloc((void**)&d_driven, numNeurons * sizeof(float));
	cudaMalloc((void**)&d_cf, numNeurons * sizeof(bool));
	cudaMalloc((void**)&d_PSPF, numNeurons * sizeof(float));
	cudaMalloc((void**)&d_PVR, numNeurons * sizeof(float));


	cudaMalloc((void**)&d_source, numEdges * sizeof(int));
	cudaMalloc((void**)&d_target, numEdges * sizeof(int));
	cudaMalloc((void**)&d_weight, numEdges * sizeof(float));
	cudaMalloc((void**)&d_delay, numEdges*sizeof(float));


	cudaMemcpy(d_v, h_v, numNeurons * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_u, h_u, numNeurons * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_I, h_I, numNeurons * sizeof(float), cudaMemcpyHostToDevice);


	cudaMemcpy(d_source, h_source.data(), numEdges * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_target, h_target.data(), numEdges * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_weight, h_weight.data(), numEdges * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_delay, h_delay.data(), numEdges * sizeof(float), cudaMemcpyHostToDevice);

	/* Generate Input Patterns */
	std::vector<int> h_enumeration;
	for (int i = 0; i < numNeurons; i++)
	{
		h_enumeration.push_back(i);
	}

	std::random_shuffle(h_enumeration.begin(), h_enumeration.end());
	


	/* Run Simulation */
	for (int num = 0; num < numPatterns; num++)
	{

		std::cout << "Pattern number " << num << std::endl;


		for (int i = 0; i < numNeurons; i++)
		{
			h_driven[i] = 0;

		}

		for (int i = 0; i < numDriven; i++)
		{
			int idx = h_enumeration[(i + num*offset)%numNeurons];
			h_driven[idx] = valDriven;
		}

		cudaMemcpy(d_driven, h_driven, numNeurons * sizeof(float), cudaMemcpyHostToDevice);


		for (int t = 0; t < equilizationTime; t++)
		{
			/* Run Timesteps, No Communication */
			NeuronTimestepNoWrite << <(numNeurons + numThreads - 1) / numThreads, numThreads >> >(
				numNeurons,
				numExcit,
				d_v,
				d_u,
				d_I,
				d_cf,
				d_driven,
				t,
				maxDelay);
		}

		for (int t = 0; t < transientTime; t++)
		{
			/* Run Timesteps, Communication, No Writing */
			NeuronTimestepNoWrite << <(numNeurons + numThreads - 1) / numThreads, numThreads >> >(
				numNeurons,
				numExcit,
				d_v,
				d_u,
				d_I,
				d_cf,
				d_driven,
				t,
				maxDelay);

			CommunicationPhase << <(numEdges + numThreads - 1) / numThreads, numThreads >> >(
				numEdges,
				d_cf,
				d_source,
				d_target,
				d_weight,
				d_I,
				t,
				maxDelay);

		}

		for (int t = 0; t < T; t++)
		{
			/* Run Timesteps, Communication, Write Results*/
			NeuronTimestepWrite << <(numNeurons + numThreads - 1) / numThreads, numThreads >> >(
				numNeurons,
				numExcit,
				d_v,
				d_u,
				d_I,
				d_cf,
				d_driven,
				t,
				maxDelay,
				d_PSPF,
				d_PVR,
				normalizer);

			CommunicationPhase << <(numEdges + numThreads - 1) / numThreads, numThreads >> >(
				numEdges,
				d_cf,
				d_source,
				d_target,
				d_weight,
				d_I,
				t,
				maxDelay);

			cudaMemcpy(SpikeTrainYard[t], d_cf, numNeurons * sizeof(bool), cudaMemcpyDeviceToHost);
			cudaMemcpy(VoltageTrace[t], d_v, numNeurons * sizeof(float), cudaMemcpyDeviceToHost);
			cudaMemcpy(PSPFs[num][t], d_PSPF, numNeurons * sizeof(float), cudaMemcpyDeviceToHost);
			cudaMemcpy(PVRs[num][t], d_PVR, numNeurons * sizeof(float), cudaMemcpyDeviceToHost);


			/* Reset Neurons */
			cudaMemcpy(d_v, h_v, numNeurons * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(d_u, h_u, numNeurons * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(d_I, h_I, numNeurons * sizeof(float), cudaMemcpyHostToDevice);
		}

	}
	/* Analyzing Run */
	
	std::cout << "Analyzing Run" << std::endl;
	
	/* PSPM Values */
	

	
	float value;

	float decayTable[PSPFCutoff];
	for (int i = 0; i < PSPFCutoff;i++)
	{
		decayTable[i] = exp(-(i+1)/tauPSPF);
	}

	for (int num = 0; num < numPatterns; num++)
	{
		for (int t = T - 1; t >= 0; t--)
		{
			for (int n = 0; n < numNeurons; n++)
			{
				value = PSPFs[num][t][n];
				for (int h = 1; h < PSPFCutoff + 1; h++)
				{
					if (t + h < T)
					{
						PSPFs[num][t + h][n] += value * decayTable[h - 1];
					}
				}
			}
		}
	}


	/* Population Van Rossum Values */

	for (int num = 0; num < numPatterns; num++)
	{
		for (int t = T - 1; t >= 0; t--)
		{
			for (int n = 0; n < numNeurons; n++)
			{
				value = PVRs[num][t][n];
				for (int h = 1; h < PSPFCutoff + 1; h++)
				{
					if (t + h < T)
					{
						PVRs[num][t + h][n] += value * decayTable[h - 1];
					}
				}
			}
		}
	}

	std::cout << "Writing Raw Data" << std::endl;

	std::ofstream rawPSPMdata;
	std::ofstream rawPVRdata;

	if (writeRawData)
	{
		rawPSPMdata.open("rawPSPMdata.xml");
		rawPVRdata.open("rawPVRdata.xml");
		for (int num = 0; num < numPatterns; num++)
		{
			rawPSPMdata << "<pattern>\n";
			rawPSPMdata << "  <id>" << num << "</id>\n";
			rawPVRdata << "<pattern>\n";
			rawPVRdata << "  <id>" << num << "</id>\n";
			for (int n = 0; n < numNeurons; n++)
			{
				rawPSPMdata << "  <neuron>\n    <nid>" << n << "</nid>\n";
				rawPVRdata << "  <neuron>\n    <nid>" << n << "</nid>\n";
				for (int t = 0; t < T; t++)
				{
					rawPSPMdata << "      <infotuple>(" << t << "," << PSPFs[num][t][n] << ")</infotuple>\n";
					rawPVRdata << "      <infotuple>(" << t << "," << PVRs[num][t][n] << ")</infotuple>\n";
				}
				rawPSPMdata << "  </neuron>\n";
				rawPVRdata << "  </neuron>\n";
			}
			rawPSPMdata << "</pattern>\n\n";
			rawPVRdata << "</pattern>\n\n";
		}

		rawPSPMdata.close();
		rawPVRdata.close();
	}

	std::cout << "Done Writing Raw Data, Calculating Distances" << std::endl;

	/* Code To Write Raw Data */

	std::ofstream distanceData;
	
	/* Calculate Distances */
	distanceData.open("entire_network_distance.csv");

	/* PSPM */

	std::cout << "PSPM" << std::endl;

	float **PSPM = new float*[numPatterns];
	for (int i = 0; i < numPatterns; i++)
	{
		PSPM[i] = new float[numPatterns];
	}

	float hold = 0;

	for (int N = 0; N < numPatterns; N++)
	{
		PSPM[N][N] = 0;
		for (int M = N+1; M < numPatterns; M++)
		{
			hold = 0;
			for (int n = 0; n < numNeurons; n++)
			{
				for (int t = 0; t < T; t++)
				{
					hold += pow((PSPFs[N][t][n] - PSPFs[M][t][n]), 2);
				}
			}
			hold /= tauPSPF;
			PSPM[N][M] = sqrt(hold);
			PSPM[M][N] = sqrt(hold);
		}
	}


	std::cout << "Population Van Rossum" << std::endl;
	/* Population Van Rossum */

	float **PVR = new float*[numPatterns];
	for (int i = 0; i < numPatterns; i++)
	{
		PVR[i] = new float[numPatterns];
	}

	hold = 0;

	for (int N = 0; N < numPatterns; N++)
	{
		PVR[N][N] = 0;
		for (int M = N + 1; M < numPatterns; M++)
		{
			hold = 0;
			for (int n = 0; n < numNeurons; n++)
			{
				for (int t = 0; t < T; t++)
				{
					hold += pow((PVRs[N][t][n] - PVRs[M][t][n]), 2);
				}
			}
			hold /= tauPSPF;
			PVR[N][M] = hold;
			PVR[M][N] = hold;

			/* Write PSPM Distance */

			distanceData << PSPM[N][M] << "," << PVR[N][M] << "\n";



		}
	}

	distanceData.close();

	std::cout << "Done With Program, Cleaning Up Reserved Memory" << std::endl;

	/* Clean Up Code */

	cudaDeviceReset();

	for (int t = 0; t < T; t++)
	{
		delete[] SpikeTrainYard[t];
		delete[] VoltageTrace[t];
	}

	delete[] h_v; delete[] h_u; delete[] h_I; delete[] h_cf; delete[] SpikeTrainYard; delete[] h_driven;
	delete[] VoltageTrace;


	for (int i = 0; i < numPatterns; i++)
	{
		for (int t = 0; t < T; t++)
		{
			delete[] PSPFs[i][t];
			delete[] PVRs[i][t];
		}
		delete[] PSPFs[i];
		delete[] PSPM[i];
		delete[] PVRs[i];
		delete[] PVR[i];
	}
	delete[] PSPFs;
	delete[] PSPM;
	delete[] PVRs;
	delete[] PVR;

	return 0;
}