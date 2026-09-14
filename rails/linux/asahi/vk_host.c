/* Optional host. Does not create a VkInstance in Home automations.
 * Link only on an Asahi box: cc vk_host.c -lvulkan
 */
#include <stdio.h>
int junior_vk_available(void) {
#ifdef JUNIOR_VK
  return 1;
#else
  return 0;
#endif
}
int main(void) {
  printf("vk_available=%d dispatch=0\n", junior_vk_available());
  return 0;
}
