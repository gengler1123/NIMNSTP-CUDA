#ifndef __KERNELS_CUH__
#define __KERNELS_CUH__


__global__ void NeuronTimestepWrite(
	int numNeur,
	int numExcit,
	float *d_v,
	float *d_u,
	float *d_I,
	bool *d_cf,
	float *d_driven,
	int t,
	int maxDelay,
	float *d_PSPF,
	float *d_PVR,
	float normalizer)
{

	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < numExcit)
	{
		int index = (i * maxDelay) + (t % maxDelay);
		d_cf[i] = false;
		d_PSPF[i] = d_I[index]/normalizer;
		d_PVR[i] = 0;
		for (int dt = 0; dt < 4; dt++)
		{
			float dv = (0.7 * (d_v[i] + 60)*(d_v[i] + 40) - d_u[i] + d_I[index] + d_driven[i]) / 100;
			float du = (0.03 * (-2 * (d_v[i] + 60) - d_u[i]));
			d_v[i] += 0.25*dv;
			d_u[i] += 0.25*du;

			if (d_v[i] > 35)
			{
				d_cf[i] = true;
				d_PVR[i] = 1;
				d_v[i] = -50;
				d_u[i] += 100;
				break;
			}
			d_I[index] = 0;
		}


	}
	else if (i < numNeur)
	{
		int index = (i * maxDelay) + (t % maxDelay);
		d_PSPF[i] = d_I[index];
		d_cf[i] = false;
		d_PVR[i] = 0;
		for (int dt = 0; dt < 4; dt++)
		{
			float dv = (1.2 * (d_v[i] + 75)*(d_v[i] + 45) - d_u[i] + d_I[index] + d_driven[i]) / 150;
			float du = (0.01 * (5 * (d_v[i] + 75) - d_u[i]));
			d_v[i] += 0.25*dv;
			d_u[i] += 0.25*du;

			if (d_v[i] > 50)
			{
				d_cf[i] = true;
				d_PVR[i] = 1;
				d_v[i] = -56;
				d_u[i] += 130;
				break;
			}
		}

		d_I[index] = 0;
	}

}


__global__ void NeuronTimestepNoWrite(
	int numNeur,
	int numExcit,
	float *d_v,
	float *d_u,
	float *d_I,
	bool *d_cf,
	float *d_driven,
	int t,
	int maxDelay)
{

	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < numExcit)
	{
		int index = (i * maxDelay) + (t % maxDelay);
		d_cf[i] = false;
		for (int dt = 0; dt < 4; dt++)
		{
			float dv = (0.7 * (d_v[i] + 60)*(d_v[i] + 40) - d_u[i] + d_I[index] + d_driven[i]) / 100;
			float du = (0.03 * (-2 * (d_v[i] + 60) - d_u[i]));
			d_v[i] += 0.25*dv;
			d_u[i] += 0.25*du;

			if (d_v[i] > 35)
			{
				d_cf[i] = true;
				d_v[i] = -50;
				d_u[i] += 100;
				break;
			}
			d_I[index] = 0;
		}


	}
	else if (i < numNeur)
	{
		int index = (i * maxDelay) + (t % maxDelay);

		d_cf[i] = false;
		for (int dt = 0; dt < 4; dt++)
		{
			float dv = (1.2 * (d_v[i] + 75)*(d_v[i] + 45) - d_u[i] + d_I[index] + d_driven[i]) / 150;
			float du = (0.01 * (5 * (d_v[i] + 75) - d_u[i]));
			d_v[i] += 0.25*dv;
			d_u[i] += 0.25*du;

			if (d_v[i] > 50)
			{
				d_cf[i] = true;
				d_v[i] = -56;
				d_u[i] += 130;
				break;
			}
		}

		d_I[index] = 0;
	}

}



__global__ void CommunicationPhase(
	int numEdge,
	bool *d_cf,
	int *d_source,
	int *d_target,
	float *d_weight,
	float *d_I,
	int t,
	int maxDelay)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < numEdge)
	{
		if (d_cf[d_source[i]])
		{
			int index = (d_target[i] * maxDelay) + (t % maxDelay);
			atomicAdd(&d_I[index], d_weight[i]);
		}
	}
}


__global__ void calculateDistance(
	int size,
	float *d_PSPF0,
	float *d_PSPF1,
	int T,
	float *d_PSPM)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i >= size)
	{
		return;
	}

	for (int idx = 0; idx < T; idx++)
	{
		d_PSPM[i] += pow((d_PSPF0[idx] - d_PSPF1[idx]),2);
	}

	d_PSPM[i] = sqrt(d_PSPM[i]);
}


#endif