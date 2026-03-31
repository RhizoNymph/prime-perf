#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* Linked-list node for chaining */
typedef struct Node {
    char *key;
    int key_len;
    char *val;
    int val_len;
    struct Node *next;
} Node;

/* FNV-1a hash */
static uint32_t fnv1a(const char *data, int len) {
    uint32_t h = 2166136261u;
    for (int i = 0; i < len; i++) {
        h ^= (uint8_t)data[i];
        h *= 16777619u;
    }
    return h;
}

/* Next power of 2 >= n */
static int next_pow2(int n) {
    int p = 1;
    while (p < n) p <<= 1;
    return p;
}

int main(void) {
    int32_t n_insert;
    if (fread(&n_insert, sizeof(int32_t), 1, stdin) != 1) return 1;

    int capacity = next_pow2(n_insert < 16 ? 16 : n_insert);
    int mask = capacity - 1;

    Node **buckets = calloc(capacity, sizeof(Node *));
    if (!buckets) return 1;

    /* Insert phase */
    for (int i = 0; i < n_insert; i++) {
        int32_t key_len;
        if (fread(&key_len, sizeof(int32_t), 1, stdin) != 1) return 1;
        char *key = malloc(key_len);
        if (!key) return 1;
        if ((int)fread(key, 1, key_len, stdin) != key_len) return 1;

        int32_t val_len;
        if (fread(&val_len, sizeof(int32_t), 1, stdin) != 1) return 1;
        char *val = malloc(val_len);
        if (!val) return 1;
        if ((int)fread(val, 1, val_len, stdin) != val_len) return 1;

        uint32_t h = fnv1a(key, key_len) & mask;

        /* Search for existing key to update */
        Node *cur = buckets[h];
        int found = 0;
        while (cur) {
            if (cur->key_len == key_len && memcmp(cur->key, key, key_len) == 0) {
                /* Duplicate key: update value (last wins) */
                free(cur->val);
                cur->val = val;
                cur->val_len = val_len;
                free(key);
                found = 1;
                break;
            }
            cur = cur->next;
        }

        if (!found) {
            Node *node = malloc(sizeof(Node));
            if (!node) return 1;
            node->key = key;
            node->key_len = key_len;
            node->val = val;
            node->val_len = val_len;
            node->next = buckets[h];
            buckets[h] = node;
        }
    }

    /* Lookup phase */
    int32_t n_lookup;
    if (fread(&n_lookup, sizeof(int32_t), 1, stdin) != 1) return 1;

    for (int i = 0; i < n_lookup; i++) {
        int32_t key_len;
        if (fread(&key_len, sizeof(int32_t), 1, stdin) != 1) return 1;
        char key_buf[256];
        if ((int)fread(key_buf, 1, key_len, stdin) != key_len) return 1;

        uint32_t h = fnv1a(key_buf, key_len) & mask;

        Node *cur = buckets[h];
        int found = 0;
        while (cur) {
            if (cur->key_len == key_len && memcmp(cur->key, key_buf, key_len) == 0) {
                int32_t one = 1;
                fwrite(&one, sizeof(int32_t), 1, stdout);
                fwrite(&cur->val_len, sizeof(int32_t), 1, stdout);
                fwrite(cur->val, 1, cur->val_len, stdout);
                found = 1;
                break;
            }
            cur = cur->next;
        }

        if (!found) {
            int32_t zero = 0;
            fwrite(&zero, sizeof(int32_t), 1, stdout);
        }
    }

    /* Cleanup */
    for (int i = 0; i < capacity; i++) {
        Node *cur = buckets[i];
        while (cur) {
            Node *next = cur->next;
            free(cur->key);
            free(cur->val);
            free(cur);
            cur = next;
        }
    }
    free(buckets);

    return 0;
}
